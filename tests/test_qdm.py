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
