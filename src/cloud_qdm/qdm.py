"""Quantile Delta Mapping training and adjustment."""

from __future__ import annotations

import xarray as xr

from cloud_qdm.config import QDMConfig
from cloud_qdm.sources import VARIABLE_METADATA


class QDMError(RuntimeError):
    """Raised when QDM training or adjustment fails."""


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
        adjustment = QuantileDeltaMapping.train(**kwargs)
        adjustment.ds.load()
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
        corrected = adjustment.adjust(
            simulation.chunk({"time": -1}),
            interp=config.interpolation,
            extrapolation=config.extrapolation,
        )
    except Exception as exc:
        raise QDMError(f"QDM adjustment failed for {variable}: {exc}") from exc
    if variable == "pr":
        corrected = corrected.clip(min=0)
    corrected.name = variable
    corrected.attrs.update(
        {
            "units": VARIABLE_METADATA[variable]["units"],
            "bias_adjustment": "Quantile Delta Mapping",
            "qdm_group": config.group,
            "qdm_nquantiles": config.nquantiles,
        }
    )
    return corrected


def save_adjustment(adjustment, path: str) -> None:
    """Persist xsdba training parameters for provenance and reuse."""
    adjustment.ds.to_netcdf(path)
