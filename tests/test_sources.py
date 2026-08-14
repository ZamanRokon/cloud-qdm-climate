from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm.config import Bounds, Period
from cloud_qdm.sources import _fetch_ee_collection, _fetch_ee_variables


class FakeCollection:
    def __init__(self):
        self.selected = None

    def filterBounds(self, _region):
        return self

    def filterDate(self, _start, _end):
        return self

    def select(self, band):
        self.selected = band
        return self

    def map(self, _function):
        return self

    def size(self):
        return SimpleNamespace(getInfo=lambda: 1)


def test_xee_request_uses_supported_spatial_arguments(monkeypatch) -> None:
    region = object()
    fake_ee = SimpleNamespace(Geometry=SimpleNamespace(Rectangle=lambda *_args, **_kwargs: region))
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    captured = {}

    def fake_open_dataset(_collection, **kwargs):
        captured.update(kwargs)
        return xr.Dataset(
            {
                "temperature_2m": xr.DataArray(
                    np.ones((1, 2, 2)),
                    coords={
                        "time": pd.date_range("2000-01-01", periods=1),
                        "lat": [0.0, 1.0],
                        "lon": [90.0, 91.0],
                    },
                    dims=("time", "lat", "lon"),
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    result = _fetch_ee_collection(
        FakeCollection(),
        band="temperature_2m",
        output_name="tas",
        period=Period(date(2000, 1, 1), date(2000, 1, 1), "test"),
        bounds=Bounds(90.0, 0.0, 91.0, 1.0),
        resolution_degrees=0.1,
        chunk_years=1,
        units="K",
    )

    assert result.name == "tas"
    assert captured == {
        "engine": "ee",
        "crs": "EPSG:4326",
        "scale": 0.1,
        "geometry": region,
    }


def test_multiband_request_opens_all_variables_together(monkeypatch) -> None:
    region = object()
    fake_ee = SimpleNamespace(Geometry=SimpleNamespace(Rectangle=lambda *_args, **_kwargs: region))
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    collection = FakeCollection()
    calls = {"count": 0}

    def fake_open_dataset(_collection, **_kwargs):
        calls["count"] += 1
        coords = {
            "time": pd.date_range("2000-01-01", periods=1),
            "lat": [0.0, 1.0],
            "lon": [90.0, 91.0],
        }
        return xr.Dataset(
            {
                variable: xr.DataArray(
                    np.ones((1, 2, 2)), coords=coords, dims=("time", "lat", "lon")
                )
                for variable in ("tas", "tasmax", "tasmin", "pr")
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    results = _fetch_ee_variables(
        collection,
        bands={name: name for name in ("tas", "tasmax", "tasmin", "pr")},
        period=Period(date(2000, 1, 1), date(2000, 1, 1), "test"),
        bounds=Bounds(90.0, 0.0, 91.0, 1.0),
        resolution_degrees=0.25,
        chunk_years=10,
        units={"tas": "K", "tasmax": "K", "tasmin": "K", "pr": "mm d-1"},
    )

    assert calls["count"] == 1
    assert collection.selected == ["tas", "tasmax", "tasmin", "pr"]
    assert set(results) == {"tas", "tasmax", "tasmin", "pr"}
