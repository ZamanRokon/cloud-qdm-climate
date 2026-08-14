"""Quantile Delta Mapping training and adjustment."""

from __future__ import annotations

import zlib
from contextlib import contextmanager

import numpy as np
import xarray as xr

from cloud_qdm.config import QDMConfig
from cloud_qdm.sources import VARIABLE_METADATA


class QDMError(RuntimeError):
    """Raised when QDM training or adjustment fails."""


def _stable_seed(config: QDMConfig, variable: str, data: xr.DataArray, operation: str) -> int:
    """Derive a repeatable seed for one variable, period, and operation."""
    start = str(data["time"].values[0])
    end = str(data["time"].values[-1])
    token = f"{variable}:{operation}:{start}:{end}:{tuple(data.sizes.items())}"
    return zlib.crc32(token.encode("utf-8"), config.random_seed)


@contextmanager
def _numpy_seed(seed: int):
    """Temporarily seed NumPy without changing the caller's random state."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def train_qdm(
    reference: xr.DataArray,
    historical: xr.DataArray,
    *,
    variable: str,
    config: QDMConfig,
):
    """Train one monthly, per-grid-cell QDM adjustment object."""
    try:
        from xsdba.adjustment import QuantileDeltaMapping
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise QDMError("xsdba is required; install the 'cloud' extra.") from exc

    metadata = VARIABLE_METADATA[variable]
    units = str(metadata["units"])
    reference = reference.copy()
    historical = historical.copy()
    reference.attrs["units"] = units
    historical.attrs["units"] = units

    kwargs: dict[str, object] = {
        "ref": reference.chunk({"time": -1}),
        "hist": historical.chunk({"time": -1}),
        "nquantiles": config.nquantiles,
        "group": config.group,
        "kind": metadata["qdm_kind"],
    }
    if variable == "pr" and config.adapt_wet_day_frequency:
        kwargs["adapt_freq_thresh"] = f"{config.wet_day_threshold_mm} mm d-1"

    try:
        seed = _stable_seed(config, variable, historical, "train")
        with _numpy_seed(seed):
            adjustment = QuantileDeltaMapping.train(**kwargs)
            adjustment.ds.load()
        adjustment.ds.attrs["qdm_random_seed"] = seed
        return adjustment
    except Exception as exc:
        raise QDMError(f"QDM training failed for {variable}: {exc}") from exc


def apply_qdm(
    adjustment,
    simulation: xr.DataArray,
    *,
    variable: str,
    config: QDMConfig,
) -> xr.DataArray:
    """Apply a trained adjustment and enforce variable-domain bounds."""
    simulation = simulation.copy()
    simulation.attrs["units"] = str(VARIABLE_METADATA[variable]["units"])
    try:
        seed = _stable_seed(config, variable, simulation, "adjust")
        with _numpy_seed(seed):
            corrected = adjustment.adjust(
                simulation.chunk({"time": -1}),
                interp=config.interpolation,
                extrapolation=config.extrapolation,
            )
    except Exception as exc:
        raise QDMError(f"QDM adjustment failed for {variable}: {exc}") from exc
    if variable == "pr":
        finite_input = np.isfinite(simulation)
        corrected = xr.where(
            finite_input,
            xr.where(np.isfinite(corrected), corrected, 0.0),
            np.nan,
        ).clip(min=0)
    else:
        corrected = corrected.where(np.isfinite(corrected))
    corrected.name = variable
    corrected.attrs.update(
        {
            "units": VARIABLE_METADATA[variable]["units"],
            "bias_adjustment": "Quantile Delta Mapping",
            "qdm_group": config.group,
            "qdm_nquantiles": config.nquantiles,
            "qdm_random_seed": seed,
        }
    )
    return corrected


def save_adjustment(adjustment, path: str) -> None:
    """Persist xsdba training parameters for provenance and reuse."""
    adjustment.ds.to_netcdf(path)
