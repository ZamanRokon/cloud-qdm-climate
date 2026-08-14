"""Coordinate normalization, subsetting, regridding, and physical constraints."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cloud_qdm.config import Bounds


class DataAlignmentError(ValueError):
    """Raised when reference and model data cannot be aligned safely."""


def normalize_coordinates(
    data: xr.DataArray,
    *,
    latitude_name: str = "lat",
    longitude_name: str = "lon",
    time_name: str = "time",
) -> xr.DataArray:
    """Rename common coordinates and normalize longitude to [-180, 180)."""
    rename: dict[str, str] = {}
    for source, target in (
        (latitude_name, "lat"),
        (longitude_name, "lon"),
        (time_name, "time"),
    ):
        if source != target and (source in data.dims or source in data.coords):
            rename[source] = target
    out = data.rename(rename)
    missing = [name for name in ("time", "lat", "lon") if name not in out.coords]
    if missing:
        raise DataAlignmentError("Missing required coordinate(s): " + ", ".join(missing))

    longitude = ((out["lon"] + 180) % 360) - 180
    out = out.assign_coords(lon=longitude).sortby("lon").sortby("lat").sortby("time")

    time_values = np.asarray(out["time"].values)
    _, unique_indices = np.unique(time_values, return_index=True)
    if len(unique_indices) != len(time_values):
        out = out.isel(time=np.sort(unique_indices))
    return out.transpose("time", "lat", "lon")


def subset_bounds(data: xr.DataArray, bounds: Bounds) -> xr.DataArray:
    """Select a rectangular EPSG:4326 AOI from normalized coordinates."""
    out = data.sel(
        lon=slice(bounds.min_lon, bounds.max_lon),
        lat=slice(bounds.min_lat, bounds.max_lat),
    )
    if out.sizes.get("lat", 0) == 0 or out.sizes.get("lon", 0) == 0:
        raise DataAlignmentError("AOI does not intersect the supplied data coordinates.")
    return out


def regrid_to_reference(
    model: xr.DataArray,
    reference: xr.DataArray,
    *,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate a model field to a reference grid without inventing fill values."""
    source_attrs = dict(model.attrs)
    out = (
        model.sortby("lat")
        .sortby("lon")
        .interp(lat=reference["lat"], lon=reference["lon"], method=method)
    )
    out.attrs.update(source_attrs)
    return out.transpose("time", "lat", "lon")


def align_calibration_time(
    reference: xr.DataArray,
    historical: xr.DataArray,
    *,
    minimum_coverage: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align calibration dates and reject incomplete overlap."""
    reference_count = reference.sizes.get("time", 0)
    historical_count = historical.sizes.get("time", 0)
    reference, historical = xr.align(reference, historical, join="inner")
    common = reference.sizes.get("time", 0)
    denominator = max(reference_count, historical_count, 1)
    coverage = common / denominator
    if common == 0 or coverage < minimum_coverage:
        raise DataAlignmentError(
            f"Calibration time coverage is {coverage:.1%}; required {minimum_coverage:.1%}."
        )
    return reference, historical


def enforce_temperature_ordering(
    tas: xr.DataArray,
    tasmax: xr.DataArray,
    tasmin: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Enforce tasmin <= tas <= tasmax at every valid cell and timestep."""
    tas, tasmax, tasmin = xr.align(tas, tasmax, tasmin, join="exact")
    stacked = xr.concat([tasmin, tas, tasmax], dim="temperature_member")
    low = stacked.min(dim="temperature_member", skipna=False)
    high = stacked.max(dim="temperature_member", skipna=False)
    middle = stacked.sum(dim="temperature_member", skipna=False) - low - high

    low.attrs.update(tasmin.attrs)
    middle.attrs.update(tas.attrs)
    high.attrs.update(tasmax.attrs)
    low.name, middle.name, high.name = "tasmin", "tas", "tasmax"
    return {"tasmin": low, "tas": middle, "tasmax": high}
