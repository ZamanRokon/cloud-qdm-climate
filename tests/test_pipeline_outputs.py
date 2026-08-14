from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import array as da
from dask import delayed

from cloud_qdm import pipeline
from cloud_qdm.config import Period, load_config
from cloud_qdm.pipeline import _merge_future_segments, _save_period
from cloud_qdm.reporting import save_netcdf, save_netcdf_batch, summarize
from cloud_qdm.sources import VARIABLES


def test_model_fetch_is_padded_beyond_the_target_aoi(monkeypatch) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    config = load_config(source)
    captured = {}

    def fake_fetch_gddp(**kwargs):
        captured.update(kwargs)
        return "model-data"

    monkeypatch.setattr(pipeline, "fetch_gddp", fake_fetch_gddp)
    result = pipeline._fetch_model(
        config,
        model="ACCESS-CM2",
        scenario="ssp245",
        period=config.future_windows[0],
        variable="pr",
    )

    assert result == "model-data"
    assert captured["bounds"].as_list() == [88.5, 22.5, 93.5, 27.0]


def test_netcdf_writer_never_serializes_infinity(tmp_path: Path) -> None:
    data = xr.DataArray(
        np.array([0.0, np.nan, np.inf, 2.0], dtype=np.float32),
        dims="time",
        coords={"time": pd.date_range("2000-01-01", periods=4)},
        name="pr",
        attrs={"units": "mm d-1"},
    )
    path = save_netcdf(data, tmp_path / "pr.nc", {})
    row = summarize(
        data,
        model="TEST",
        scenario="ssp245",
        period="test",
        variable="pr",
        output_path=path,
    )

    with xr.open_dataarray(path) as saved:
        assert not np.isinf(saved.values).any()
    assert row["non_finite_values"] == 2
    assert row["finite_fraction"] == 0.5
    assert row["mean"] == 1.0


def test_batch_writer_computes_a_shared_graph_once(tmp_path: Path) -> None:
    calls = {"count": 0}

    @delayed(pure=True)
    def source_values():
        calls["count"] += 1
        return np.arange(4, dtype=np.float32)

    shared = da.from_delayed(source_values(), shape=(4,), dtype=np.float32)
    items = []
    for index, variable in enumerate(("tas", "tasmax", "tasmin")):
        data = xr.DataArray(shared + index, dims="time", name=variable)
        items.append((data, tmp_path / f"{variable}.nc", {}))

    paths = save_netcdf_batch(items)

    assert len(paths) == 3
    assert calls["count"] == 1
    assert all(path.exists() for path in paths)


def test_batch_writer_rejects_different_spatial_grids(tmp_path: Path) -> None:
    coarse = xr.DataArray(
        np.ones((1, 2)),
        dims=("time", "lat"),
        coords={"time": [0], "lat": [0.0, 1.0]},
        name="tas",
    )
    fine = xr.DataArray(
        np.ones((1, 3)),
        dims=("time", "lat"),
        coords={"time": [0], "lat": [0.0, 0.5, 1.0]},
        name="pr",
    )

    with pytest.raises(ValueError, match="exactly the same coordinates"):
        save_netcdf_batch([(coarse, tmp_path / "tas.nc", {}), (fine, tmp_path / "pr.nc", {})])


def test_period_writer_preserves_separate_temperature_and_precipitation_grids(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    base = load_config(source)
    config = replace(
        base,
        name="grid-test",
        output_dir=str(tmp_path),
        processing=replace(base.processing, scratch_dir=str(tmp_path / "scratch")),
    )
    time = pd.date_range("2000-01-01", periods=2)
    corrected = {
        name: xr.DataArray(
            np.ones((2, 2, 1)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": [0.0, 1.0], "lon": [90.0]},
            name=name,
        )
        for name in ("tas", "tasmax", "tasmin")
    }
    corrected["pr"] = xr.DataArray(
        np.ones((2, 3, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0, 0.5, 1.0], "lon": [90.0]},
        name="pr",
    )

    paths = _save_period(
        config,
        model="TEST",
        scenario="historical",
        period=Period(time[0].date(), time[-1].date(), "test"),
        corrected=corrected,
        summary_rows=[],
        record_summary=False,
    )

    sizes = {}
    for path in paths:
        with xr.open_dataarray(path) as saved:
            sizes[saved.name] = saved.sizes["lat"]
    assert sizes == {"tas": 2, "tasmax": 2, "tasmin": 2, "pr": 3}


def test_future_segments_merge_to_continuous_2015_2100_outputs(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    base = load_config(source)
    config = replace(
        base,
        name="merge-test",
        output_dir=str(tmp_path),
        processing=replace(base.processing, scratch_dir=str(tmp_path / "scratch")),
    )
    segment_root = config.segment_dir
    segment_paths: dict[str, list[Path]] = {variable: [] for variable in VARIABLES}

    for period in config.future_windows:
        time = pd.date_range(period.start, period.end, freq="D")
        for index, variable in enumerate(VARIABLES):
            path = segment_root / "TEST" / "ssp245" / period.label / f"{variable}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = np.full((len(time), 1, 1), index, dtype=np.float32)
            xr.DataArray(
                values,
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": [23.0], "lon": [91.0]},
                name=variable,
                attrs={"units": "mm d-1" if variable == "pr" else "K"},
            ).to_netcdf(path)
            segment_paths[variable].append(path)

    summary_rows: list[dict[str, object]] = []
    outputs = _merge_future_segments(
        config,
        model="TEST",
        scenario="ssp245",
        segment_paths=segment_paths,
        summary_rows=summary_rows,
    )

    expected_time = pd.date_range("2015-01-01", "2100-12-31", freq="D")
    assert len(outputs) == 4
    assert len(summary_rows) == 4
    assert not segment_root.exists()
    for output in map(Path, outputs):
        assert output.parent.name == "2015-2100"
        with xr.open_dataarray(output) as data:
            assert data.sizes["time"] == len(expected_time)
            assert pd.Timestamp(data["time"].values[0]) == expected_time[0]
            assert pd.Timestamp(data["time"].values[-1]) == expected_time[-1]
            assert np.isfinite(data.values).all()
