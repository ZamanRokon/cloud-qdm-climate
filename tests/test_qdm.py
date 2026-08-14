import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("xsdba")

from cloud_qdm.config import QDMConfig
from cloud_qdm.qdm import apply_qdm, train_qdm


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
