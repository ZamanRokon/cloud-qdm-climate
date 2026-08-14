import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("xsdba")

from cloud_qdm.config import QDMConfig
from cloud_qdm.qdm import _numpy_seed, _stable_seed, apply_qdm, train_qdm


def test_temperature_qdm_reduces_simple_additive_bias() -> None:
    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    seasonal = 280 + 8 * np.sin(2 * np.pi * time.dayofyear.to_numpy() / 365.25)
    reference = xr.DataArray(
        seasonal[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [24.0], "lon": [90.0]},
        attrs={"units": "K"},
    )
    historical = reference + 3.0
    historical.attrs["units"] = "K"
    config = QDMConfig(nquantiles=10, adapt_wet_day_frequency=False)
    adjustment = train_qdm(reference, historical, variable="tas", config=config)
    corrected = apply_qdm(adjustment, historical, variable="tas", config=config).compute()
    assert abs(float((corrected - reference).mean())) < 0.25


def test_precipitation_qdm_preserves_multiplicative_change() -> None:
    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    cycle = 5 + 3 * np.sin(2 * np.pi * time.dayofyear.to_numpy() / 365.25)
    reference = xr.DataArray(
        cycle[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [24.0], "lon": [90.0]},
        attrs={"units": "mm d-1"},
    )
    historical = reference * 2
    historical.attrs["units"] = "mm d-1"
    future = historical * 1.25
    future.attrs["units"] = "mm d-1"
    config = QDMConfig(nquantiles=10, adapt_wet_day_frequency=False)

    adjustment = train_qdm(reference, historical, variable="pr", config=config)
    corrected = apply_qdm(adjustment, future, variable="pr", config=config).compute()

    expected = reference * 1.25
    assert abs(float((corrected - expected).mean())) < 0.1


def test_precipitation_output_cleans_algorithmic_non_finite_values() -> None:
    class NonFiniteAdjustment:
        @staticmethod
        def adjust(simulation: xr.DataArray, **_kwargs) -> xr.DataArray:
            return xr.DataArray(
                np.array([np.nan, np.inf, -1.0, np.inf])[:, None, None],
                dims=simulation.dims,
                coords=simulation.coords,
            )

    simulation = xr.DataArray(
        np.array([1.0, 2.0, 3.0, np.nan])[:, None, None],
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=4),
            "lat": [24.0],
            "lon": [90.0],
        },
        attrs={"units": "mm d-1"},
    )

    corrected = apply_qdm(
        NonFiniteAdjustment(),
        simulation,
        variable="pr",
        config=QDMConfig(),
    ).compute()

    np.testing.assert_array_equal(corrected.values[:3, 0, 0], [0.0, 0.0, 0.0])
    assert np.isnan(corrected.values[3, 0, 0])


def test_qdm_seed_is_stable_without_leaking_numpy_state() -> None:
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    data = xr.DataArray(
        np.ones((10, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [24.0], "lon": [90.0]},
    )
    config = QDMConfig(random_seed=123)
    seed = _stable_seed(config, "pr", data, "train")
    assert seed == _stable_seed(config, "pr", data, "train")

    np.random.seed(9)
    expected = np.random.random()
    np.random.seed(9)
    with _numpy_seed(seed):
        first = np.random.random()
    observed = np.random.random()
    with _numpy_seed(seed):
        second = np.random.random()
    assert first == second
    assert observed == expected


def test_wet_day_frequency_training_is_reproducible() -> None:
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    day = np.arange(len(time))
    reference = xr.DataArray(
        np.where(day % 3 == 0, 2.0, 0.0)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [24.0], "lon": [90.0]},
        attrs={"units": "mm d-1"},
    )
    historical = xr.DataArray(
        np.where(day % 7 == 0, 4.0, 0.0)[:, None, None],
        dims=reference.dims,
        coords=reference.coords,
        attrs={"units": "mm d-1"},
    )
    config = QDMConfig(nquantiles=10, random_seed=2026)

    first = train_qdm(reference, historical, variable="pr", config=config)
    second = train_qdm(reference, historical, variable="pr", config=config)
    different_seed = train_qdm(
        reference,
        historical,
        variable="pr",
        config=QDMConfig(nquantiles=10, random_seed=2027),
    )

    xr.testing.assert_identical(first.ds, second.ds)
    assert not first.ds.identical(different_seed.ds)
