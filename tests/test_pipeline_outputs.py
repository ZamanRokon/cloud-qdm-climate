from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm import pipeline
from cloud_qdm.config import load_config
from cloud_qdm.pipeline import _merge_future_segments
from cloud_qdm.reporting import save_netcdf, summarize
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


def test_future_segments_merge_to_continuous_2015_2100_outputs(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    config = replace(load_config(source), name="merge-test", output_dir=str(tmp_path))
    segment_root = config.run_dir / ".segments"
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
