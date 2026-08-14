import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cloud_qdm.config import Bounds
from cloud_qdm.coordinates import (
    DataAlignmentError,
    align_calibration_time,
    enforce_temperature_ordering,
    normalize_coordinates,
    regrid_to_reference,
    subset_bounds,
)


def cube(values, *, lat=(0.0, 1.0), lon=(90.0, 91.0), periods=2, name="tas"):
    latitude = np.asarray(lat, dtype=float)
    longitude = np.asarray(lon, dtype=float)
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(periods, len(latitude), len(longitude)),
        coords={
            "time": pd.date_range("2000-01-01", periods=periods),
            "lat": latitude,
            "lon": longitude,
        },
        dims=("time", "lat", "lon"),
        name=name,
        attrs={"units": "K"},
    )


def test_longitudes_are_normalized_and_bounds_subset() -> None:
    data = cube(np.arange(8), lon=(270.0, 271.0))
    normalized = normalize_coordinates(data)
    assert normalized.lon.values.tolist() == [-90.0, -89.0]
    selected = subset_bounds(normalized, Bounds(-90, 0, -89, 1))
    assert selected.shape == (2, 2, 2)


def test_regrid_preserves_missing_values() -> None:
    source = cube([1, 2, 3, np.nan, 2, 3, 4, np.nan])
    reference = cube(np.zeros(18), lat=(0.0, 0.5, 1.0), lon=(90.0, 90.5, 91.0), periods=2)
    result = regrid_to_reference(source, reference)
    assert result.shape == reference.shape
    assert np.isnan(result.isel(time=0, lat=-1, lon=-1))


def test_temperature_ordering() -> None:
    tas = cube([5, 5, 5, 5, 5, 5, 5, 5], name="tas")
    tasmin = cube([7, 3, 4, 4, 7, 3, 4, 4], name="tasmin")
    tasmax = cube([6, 4, 8, 8, 6, 4, 8, 8], name="tasmax")
    result = enforce_temperature_ordering(tas, tasmax, tasmin)
    assert bool((result["tasmin"] <= result["tas"]).all())
    assert bool((result["tas"] <= result["tasmax"]).all())


def test_alignment_rejects_low_coverage() -> None:
    reference = cube(np.arange(8))
    historical = reference.isel(time=slice(0, 1))
    with pytest.raises(DataAlignmentError, match="coverage"):
        align_calibration_time(reference, historical, minimum_coverage=0.9)
