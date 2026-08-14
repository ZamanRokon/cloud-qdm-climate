"""Earth Engine and user-supplied MSWEP data access."""

from __future__ import annotations

import glob
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm.config import Bounds, MSWEPConfig, Period
from cloud_qdm.coordinates import normalize_coordinates, subset_bounds

VARIABLES = ("tas", "tasmax", "tasmin", "pr")
VARIABLE_METADATA = {
    "tas": {
        "era5_band": "temperature_2m",
        "units": "K",
        "qdm_kind": "+",
        "reference_resolution": 0.1,
    },
    "tasmax": {
        "era5_band": "temperature_2m_max",
        "units": "K",
        "qdm_kind": "+",
        "reference_resolution": 0.1,
    },
    "tasmin": {
        "era5_band": "temperature_2m_min",
        "units": "K",
        "qdm_kind": "+",
        "reference_resolution": 0.1,
    },
    "pr": {
        "chirps_band": "precipitation",
        "units": "mm d-1",
        "qdm_kind": "*",
        "reference_resolution": 0.05,
    },
}


class DataSourceError(RuntimeError):
    """Raised when a configured data source cannot provide the requested data."""


def initialize_earth_engine(project_id: str) -> None:
    """Initialize an existing Earth Engine authentication context."""
    try:
        import ee
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise DataSourceError("Install the cloud extra: pip install -e '.[cloud]'.") from exc
    try:
        ee.Initialize(project=project_id)
    except Exception as exc:
        raise DataSourceError(
            "Earth Engine initialization failed. Run ee.Authenticate() and confirm project access."
        ) from exc


def _date_chunks(start: date, end: date, years: int) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    current = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while current <= final:
        chunk_end = min(current + pd.DateOffset(years=years) - pd.DateOffset(days=1), final)
        yield current, chunk_end
        current = chunk_end + pd.DateOffset(days=1)


def _fetch_ee_collection(
    collection: object,
    *,
    band: str,
    output_name: str,
    period: Period,
    bounds: Bounds,
    resolution_degrees: float,
    chunk_years: int,
    units: str,
) -> xr.DataArray:
    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError("earthengine-api is not installed.") from exc

    region = ee.Geometry.Rectangle(bounds.as_list(), proj="EPSG:4326", geodesic=False)
    pieces: list[xr.DataArray] = []
    for chunk_start, chunk_end in _date_chunks(period.start, period.end, chunk_years):
        filter_end = chunk_end + pd.DateOffset(days=1)
        selected = (
            collection.filterBounds(region)
            .filterDate(chunk_start.strftime("%Y-%m-%d"), filter_end.strftime("%Y-%m-%d"))
            .select(band)
            .map(lambda image: image.clip(region))
        )
        count = int(selected.size().getInfo())
        if count == 0:
            raise DataSourceError(
                f"No images for {output_name} from {chunk_start.date()} to {chunk_end.date()}."
            )
        try:
            dataset = xr.open_dataset(
                selected,
                engine="ee",
                crs="EPSG:4326",
                scale=resolution_degrees,
                geometry=region,
            )
        except Exception as exc:
            raise DataSourceError(f"xee failed while opening {output_name}: {exc}") from exc
        if band not in dataset:
            raise DataSourceError(
                f"Earth Engine response for {output_name} did not contain band '{band}'."
            )
        pieces.append(dataset[band])

    data = xr.concat(pieces, dim="time")
    data = normalize_coordinates(
        data,
        latitude_name="y" if "y" in data.dims else "lat",
        longitude_name="x" if "x" in data.dims else "lon",
    )
    data = subset_bounds(data, bounds)
    data.name = output_name
    data.attrs.update({"units": units, "source_band": band})
    return data


def fetch_era5_reference(
    *,
    collection_id: str,
    variable: str,
    period: Period,
    bounds: Bounds,
    chunk_years: int,
) -> xr.DataArray:
    """Fetch one ERA5-Land daily temperature reference."""
    if variable not in {"tas", "tasmax", "tasmin"}:
        raise DataSourceError(f"ERA5 reference is not configured for '{variable}'.")
    import ee

    metadata = VARIABLE_METADATA[variable]
    collection = ee.ImageCollection(collection_id)
    return _fetch_ee_collection(
        collection,
        band=str(metadata["era5_band"]),
        output_name=variable,
        period=period,
        bounds=bounds,
        resolution_degrees=float(metadata["reference_resolution"]),
        chunk_years=chunk_years,
        units=str(metadata["units"]),
    )


def fetch_chirps_reference(
    *,
    collection_id: str,
    period: Period,
    bounds: Bounds,
    chunk_years: int,
) -> xr.DataArray:
    """Fetch daily CHIRPS precipitation."""
    import ee

    metadata = VARIABLE_METADATA["pr"]
    return _fetch_ee_collection(
        ee.ImageCollection(collection_id),
        band=str(metadata["chirps_band"]),
        output_name="pr",
        period=period,
        bounds=bounds,
        resolution_degrees=float(metadata["reference_resolution"]),
        chunk_years=chunk_years,
        units=str(metadata["units"]),
    )


def fetch_gddp(
    *,
    collection_id: str,
    variable: str,
    model: str,
    scenario: str,
    period: Period,
    bounds: Bounds,
    chunk_years: int,
    grid_label: str | None = None,
) -> xr.DataArray:
    """Fetch one NEX-GDDP-CMIP6 model/experiment variable."""
    if variable not in VARIABLES:
        raise DataSourceError(f"Unsupported variable: {variable}")
    import ee

    collection = (
        ee.ImageCollection(collection_id)
        .filter(ee.Filter.eq("model", model))
        .filter(ee.Filter.eq("scenario", scenario))
    )
    if grid_label:
        collection = collection.filter(ee.Filter.eq("grid_label", grid_label))
    data = _fetch_ee_collection(
        collection,
        band=variable,
        output_name=variable,
        period=period,
        bounds=bounds,
        resolution_degrees=0.25,
        chunk_years=chunk_years,
        units=str(VARIABLE_METADATA[variable]["units"]),
    )
    if variable == "pr":
        data = data * 86400.0
        data.attrs["units"] = "mm d-1"
        data.attrs["unit_conversion"] = "kg m-2 s-1 multiplied by 86400"
    return data


def open_mswep(
    config: MSWEPConfig,
    *,
    period: Period,
    bounds: Bounds,
    chunks: dict[str, int] | None = None,
) -> xr.DataArray:
    """Open and subset an authorized MSWEP NetCDF collection or Zarr store."""
    path = str(Path(config.path).expanduser())
    source_chunks = "auto" if chunks else None
    try:
        if path.endswith(".zarr"):
            dataset = xr.open_zarr(path, chunks=source_chunks)
        elif any(char in path for char in "*?["):
            matches = sorted(glob.glob(path))
            if not matches:
                raise DataSourceError(f"MSWEP pattern matched no files: {path}")
            dataset = xr.open_mfdataset(matches, combine="by_coords", chunks=source_chunks)
        else:
            if not Path(path).exists():
                raise DataSourceError(f"MSWEP path does not exist: {path}")
            dataset = xr.open_dataset(path, chunks=source_chunks)
    except DataSourceError:
        raise
    except Exception as exc:
        raise DataSourceError(f"Could not open MSWEP data at {path}: {exc}") from exc

    if config.variable not in dataset:
        raise DataSourceError(
            f"MSWEP variable '{config.variable}' not found. Available: {list(dataset.data_vars)}"
        )
    data = normalize_coordinates(
        dataset[config.variable],
        latitude_name=config.latitude_name,
        longitude_name=config.longitude_name,
        time_name=config.time_name,
    )
    data = subset_bounds(data, bounds).sel(
        time=slice(period.start.isoformat(), period.end.isoformat())
    )
    if data.sizes.get("time", 0) == 0:
        raise DataSourceError("MSWEP contains no data for the calibration period.")
    data = data.astype(np.float32) * config.unit_scale

    non_finite_count = int((~np.isfinite(data)).sum().compute().item())
    if non_finite_count:
        if not config.fill_non_finite_with_zero:
            raise DataSourceError(
                f"MSWEP contains {non_finite_count:,} NaN or infinite values in the selected "
                "AOI/period. Verify that they mean dry days, then set "
                "precipitation_reference.fill_non_finite_with_zero: true to replace them."
            )
        data = xr.where(np.isfinite(data), data, 0.0)

    negative_count = int((data < 0).sum().compute().item())
    if negative_count:
        raise DataSourceError(
            f"MSWEP contains {negative_count:,} negative precipitation values after unit scaling."
        )
    if config.aggregate_to_daily:
        data = data.resample(time="1D").sum(skipna=False)
    if chunks:
        data = data.chunk(
            {
                "lat": chunks.get("lat", 40),
                "lon": chunks.get("lon", 40),
            }
        )
    data.name = "pr"
    data.attrs.update(
        {
            "units": "mm d-1",
            "source": "MSWEP (user supplied; verify version and license)",
            "unit_scale_applied": config.unit_scale,
            "non_finite_values_replaced": non_finite_count,
            "non_finite_policy": ("zero" if config.fill_non_finite_with_zero else "reject"),
        }
    )
    return data


def inspect_mswep(path: str) -> str:
    """Return a text summary of a NetCDF or Zarr store without loading arrays."""
    expanded = str(Path(path).expanduser())
    dataset = xr.open_zarr(expanded) if expanded.endswith(".zarr") else xr.open_dataset(expanded)
    return repr(dataset)
