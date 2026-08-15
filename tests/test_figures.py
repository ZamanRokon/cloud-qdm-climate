from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PIL import Image

from cloud_qdm.config import FigureConfig
from cloud_qdm.diagnostics import evaluation_rows
from cloud_qdm.figures import (
    _draw_taylor_axis,
    _pyplot,
    _save,
    make_ensemble_figures,
    make_evaluation_figures,
    make_projection_figures,
)


def _datasets(scale: float, temperature_offset: float) -> dict[str, xr.DataArray]:
    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    seasonal = np.sin(2 * np.pi * time.dayofyear.to_numpy() / 365.25)
    output = {}
    for variable in ("tas", "tasmax", "tasmin", "pr"):
        base = 5 + 2 * seasonal if variable == "pr" else 280 + 8 * seasonal
        values = np.broadcast_to(base[:, None, None], (len(time), 2, 2)).copy()
        values += np.array([[0.0, 0.2], [0.4, 0.6]])
        if variable == "pr":
            values *= scale
        else:
            values += temperature_offset
        output[variable] = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": [23.0, 24.0], "lon": [89.0, 90.0]},
            attrs={"units": "mm d-1" if variable == "pr" else "K"},
            name=variable,
        )
    return output


def test_evaluation_figure_suite_renders(tmp_path: Path) -> None:
    paths, rows = make_evaluation_figures(
        _datasets(1.0, 0.0),
        _datasets(1.4, 2.0),
        _datasets(1.05, 0.1),
        model="TEST",
        period="held-out",
        output_dir=tmp_path,
        settings=FigureConfig(enabled=True, formats=("png",), dpi=150),
        wet_day_threshold=0.1,
    )
    assert len(rows) == 8
    assert len(paths) == 11
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert (tmp_path / "skill-metrics.csv").exists()


def test_projection_and_ensemble_figure_suites_render(tmp_path: Path) -> None:
    reference = _datasets(1.0, 0.0)
    raw = _datasets(1.4, 2.0)
    corrected = _datasets(1.05, 0.1)
    future_raw = _datasets(1.68, 4.0)
    future_corrected = _datasets(1.26, 2.1)
    settings = FigureConfig(enabled=True, formats=("png",), dpi=150)

    projection_paths, projection_rows = make_projection_figures(
        raw,
        corrected,
        future_raw,
        future_corrected,
        model="TEST",
        scenario="ssp245",
        period="near-term",
        output_dir=tmp_path / "projection",
        settings=settings,
    )
    skill_rows = evaluation_rows(
        reference,
        raw,
        corrected,
        model="TEST",
        period="held-out",
    )
    ensemble_paths = make_ensemble_figures(
        skill_rows,
        projection_rows,
        output_dir=tmp_path / "core",
        settings=settings,
    )

    assert len(projection_paths) == 6
    assert len(projection_rows) == 36
    assert len(ensemble_paths) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in projection_paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in ensemble_paths)


def test_taylor_axis_is_limited_to_the_standard_positive_correlation_quadrant() -> None:
    rows = evaluation_rows(
        _datasets(1.0, 0.0),
        _datasets(1.4, 2.0),
        _datasets(1.05, 0.1),
        model="TEST",
        period="held-out",
    )
    selected = [row for row in rows if row["variable"] == "pr" and row["stage"] == "raw"]
    plt = _pyplot()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    _draw_taylor_axis(
        ax,
        selected,
        title="Precipitation — Before QDM",
        radial_max=1.75,
        point_styles={"TEST": ("#0072B2", "o")},
        style_field="model",
        show_standard_deviation_label=True,
    )

    assert np.isclose(ax.get_thetamin(), 0)
    assert np.isclose(ax.get_thetamax(), 90)
    assert np.allclose(ax.get_xlim(), (0, np.pi / 2))
    plt.close(fig)


def test_png_is_written_with_configured_600_dpi(tmp_path: Path) -> None:
    plt = _pyplot()
    fig, _ = plt.subplots(figsize=(1, 1))
    path = _save(
        fig,
        tmp_path / "publication",
        FigureConfig(enabled=True, formats=("png",), dpi=600),
    )[0]

    with Image.open(path) as image:
        assert image.info["dpi"][0] == pytest.approx(600, rel=0.01)
