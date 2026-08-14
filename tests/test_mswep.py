from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm.config import Bounds, MSWEPConfig, Period
from cloud_qdm.sources import open_mswep


def test_open_mswep_renames_scales_and_subsets(tmp_path: Path) -> None:
    path = tmp_path / "mswep.nc"
    dataset = xr.Dataset(
        {
            "rain": (
                ("date", "latitude", "longitude"),
                np.ones((4, 3, 3), dtype=np.float32),
            )
        },
        coords={
            "date": pd.date_range("2000-01-01", periods=4),
            "latitude": [22.0, 23.0, 24.0],
            "longitude": [89.0, 90.0, 91.0],
        },
    )
    dataset.to_netcdf(path)

    result = open_mswep(
        MSWEPConfig(
            path=str(path),
            variable="rain",
            latitude_name="latitude",
            longitude_name="longitude",
            time_name="date",
            unit_scale=2.0,
        ),
        period=Period(
            start=pd.Timestamp("2000-01-02").date(),
            end=pd.Timestamp("2000-01-03").date(),
            label="test",
        ),
        bounds=Bounds(89.5, 22.5, 91.0, 24.0),
    )
    assert result.dims == ("time", "lat", "lon")
    assert result.shape == (2, 2, 2)
    assert float(result.mean()) == 2.0
    assert result.attrs["units"] == "mm d-1"
