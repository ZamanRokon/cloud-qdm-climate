"""Statistics used by publication diagnostics and figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

TEMPERATURES = frozenset({"tas", "tasmax", "tasmin"})
VARIABLE_LABELS = {
    "tas": "Mean temperature",
    "tasmax": "Maximum temperature",
    "tasmin": "Minimum temperature",
    "pr": "Precipitation",
}
PLOT_UNITS = {"tas": "°C", "tasmax": "°C", "tasmin": "°C", "pr": "mm d⁻¹"}
QUANTILES = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
ETCCDI_WET_DAY_MM = 1.0


def spatial_mean(data: xr.DataArray) -> xr.DataArray:
    """Return a cosine-latitude-weighted AOI mean."""
    weights = np.cos(np.deg2rad(data["lat"])).clip(min=0)
    return data.weighted(weights).mean(("lat", "lon"), skipna=True)


def plot_units(data: xr.DataArray, variable: str) -> xr.DataArray:
    """Convert temperature to degrees Celsius for presentation."""
    return data - 273.15 if variable in TEMPERATURES else data


def paired_series(
    reference: xr.DataArray, candidate: xr.DataArray, variable: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, aligned AOI-mean daily values in presentation units."""
    reference, candidate = xr.align(
        spatial_mean(plot_units(reference, variable)),
        spatial_mean(plot_units(candidate, variable)),
        join="inner",
    )
    ref = np.asarray(reference.compute().values, dtype=float).ravel()
    sim = np.asarray(candidate.compute().values, dtype=float).ravel()
    valid = np.isfinite(ref) & np.isfinite(sim)
    return ref[valid], sim[valid]


def skill_metrics(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    *,
    variable: str,
) -> dict[str, float]:
    """Calculate normalized daily AOI-mean performance statistics."""
    ref, sim = paired_series(reference, candidate, variable)
    if ref.size < 3:
        raise ValueError(f"At least three paired values are required for {variable} metrics.")
    error = sim - ref
    ref_std = float(np.std(ref))
    sim_std = float(np.std(sim))
    correlation = float(np.corrcoef(ref, sim)[0, 1]) if ref_std and sim_std else np.nan
    centered = (sim - sim.mean()) - (ref - ref.mean())
    scale = ref_std if ref_std else np.nan
    return {
        "bias": float(error.mean()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_bias": float(error.mean() / scale),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / scale),
        "correlation": correlation,
        "std_ratio": float(sim_std / scale),
        "centered_rmse": float(np.sqrt(np.mean(centered**2)) / scale),
        "sample_size": int(ref.size),
    }


def evaluation_rows(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    *,
    model: str,
    period: str,
) -> list[dict[str, Any]]:
    """Build long-form raw/corrected evaluation metrics."""
    rows: list[dict[str, Any]] = []
    for variable in references:
        for stage, data in (("raw", raw[variable]), ("corrected", corrected[variable])):
            rows.append(
                {
                    "model": model,
                    "period": period,
                    "variable": variable,
                    "stage": stage,
                    "units": PLOT_UNITS[variable],
                    **skill_metrics(references[variable], data, variable=variable),
                }
            )
    return rows


def monthly_cycle(data: xr.DataArray, variable: str) -> np.ndarray:
    """Calculate the AOI-mean daily climatology for calendar months."""
    values = spatial_mean(plot_units(data, variable)).groupby("time.month").mean(skipna=True)
    return np.asarray(values.compute().values, dtype=float)


def quantile_values(data: xr.DataArray, variable: str) -> np.ndarray:
    """Calculate fixed quantiles of the AOI-mean daily series."""
    values = np.asarray(spatial_mean(plot_units(data, variable)).compute().values, dtype=float)
    return np.nanquantile(values, QUANTILES)


def climatology_map(data: xr.DataArray, variable: str) -> xr.DataArray:
    """Calculate a time-mean field in presentation units."""
    return plot_units(data, variable).mean("time", skipna=True).compute()


def wet_day_cycle(data: xr.DataArray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return monthly wet-day frequency and conditional wet-day intensity."""
    wet = data >= threshold
    frequency = spatial_mean(wet.groupby("time.month").mean("time") * 100)
    intensity = spatial_mean(data.where(wet).groupby("time.month").mean("time", skipna=True))
    return (
        np.asarray(frequency.compute().values, dtype=float),
        np.asarray(intensity.compute().values, dtype=float),
    )


def correlation_matrix(data: Mapping[str, xr.DataArray]) -> np.ndarray:
    """Calculate inter-variable correlations of aligned AOI-mean daily series."""
    series = [spatial_mean(plot_units(data[name], name)).rename(name) for name in VARIABLE_LABELS]
    aligned = xr.align(*series, join="inner")
    frame = pd.DataFrame(
        {
            name: np.asarray(values.compute().values, dtype=float)
            for name, values in zip(VARIABLE_LABELS, aligned, strict=True)
        }
    )
    return frame.corr().to_numpy()


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def annual_extremes(data: Mapping[str, xr.DataArray]) -> pd.DataFrame:
    """Calculate annual regional indices using the ETCCDI 1 mm wet-day rule."""
    series = {
        name: spatial_mean(plot_units(values, name)).compute() for name, values in data.items()
    }
    years = np.asarray(series["pr"]["time"].dt.year.values, dtype=int)
    rows: list[dict[str, float | int]] = []
    for year in np.unique(years):
        year_values = {
            name: np.asarray(
                values.where(values["time"].dt.year == year, drop=True).values, dtype=float
            )
            for name, values in series.items()
        }
        precipitation = year_values["pr"]
        rolling_five = pd.Series(precipitation).rolling(5, min_periods=5).sum().to_numpy()
        rows.append(
            {
                "year": int(year),
                "PRCPTOT": float(np.nansum(precipitation[precipitation >= ETCCDI_WET_DAY_MM])),
                "Rx1day": float(np.nanmax(precipitation)),
                "Rx5day": float(np.nanmax(rolling_five)),
                "CDD": float(_longest_run(precipitation < ETCCDI_WET_DAY_MM)),
                "TXx": float(np.nanmax(year_values["tasmax"])),
                "TNn": float(np.nanmin(year_values["tasmin"])),
                "DTR": float(np.nanmean(year_values["tasmax"] - year_values["tasmin"])),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def projection_change_rows(
    baseline_raw: Mapping[str, xr.DataArray],
    baseline_corrected: Mapping[str, xr.DataArray],
    future_raw: Mapping[str, xr.DataArray],
    future_corrected: Mapping[str, xr.DataArray],
    *,
    model: str,
    scenario: str,
    period: str,
) -> list[dict[str, Any]]:
    """Summarize modeled and corrected change by quantile."""
    rows: list[dict[str, Any]] = []
    for variable in VARIABLE_LABELS:
        raw_hist = quantile_values(baseline_raw[variable], variable)
        raw_future = quantile_values(future_raw[variable], variable)
        corrected_hist = quantile_values(baseline_corrected[variable], variable)
        corrected_future = quantile_values(future_corrected[variable], variable)
        if variable == "pr":
            raw_change = 100 * np.divide(
                raw_future - raw_hist,
                raw_hist,
                out=np.full_like(raw_hist, np.nan),
                where=np.abs(raw_hist) > 1e-6,
            )
            corrected_change = 100 * np.divide(
                corrected_future - corrected_hist,
                corrected_hist,
                out=np.full_like(corrected_hist, np.nan),
                where=np.abs(corrected_hist) > 1e-6,
            )
            units = "%"
        else:
            raw_change = raw_future - raw_hist
            corrected_change = corrected_future - corrected_hist
            units = "°C"
        for quantile, raw_value, corrected_value in zip(
            QUANTILES, raw_change, corrected_change, strict=True
        ):
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "period": period,
                    "variable": variable,
                    "quantile": float(quantile),
                    "units": units,
                    "raw_change": float(raw_value),
                    "corrected_change": float(corrected_value),
                    "signal_difference": float(corrected_value - raw_value),
                }
            )
    return rows
