from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm.diagnostics import (
    annual_extremes,
    evaluation_rows,
    projection_change_rows,
    skill_metrics,
)


def _field(variable: str, *, scale: float = 1.0, offset: float = 0.0) -> xr.DataArray:
    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    seasonal = np.sin(2 * np.pi * time.dayofyear.to_numpy() / 365.25)
    base = 5 + 2 * seasonal if variable == "pr" else 280 + 8 * seasonal
    values = np.broadcast_to(base[:, None, None], (len(time), 2, 2)).copy()
    values += np.array([[0.0, 0.2], [0.4, 0.6]])
    return xr.DataArray(
        values * scale + offset,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [23.0, 24.0], "lon": [89.0, 90.0]},
        attrs={"units": "mm d-1" if variable == "pr" else "K"},
        name=variable,
    )


def _datasets(scale: float, temperature_offset: float):
    return {
        variable: _field(
            variable,
            scale=scale if variable == "pr" else 1.0,
            offset=temperature_offset if variable != "pr" else 0.0,
        )
        for variable in ("tas", "tasmax", "tasmin", "pr")
    }


def test_skill_metrics_detect_bias_reduction() -> None:
    reference = _field("tas")
    raw = _field("tas", offset=3)
    corrected = _field("tas", offset=0.2)
    assert (
        skill_metrics(reference, corrected, variable="tas")["rmse"]
        < skill_metrics(reference, raw, variable="tas")["rmse"]
    )


def test_evaluation_rows_cover_each_variable_and_stage() -> None:
    reference = _datasets(1.0, 0.0)
    rows = evaluation_rows(
        reference,
        _datasets(1.4, 2.0),
        _datasets(1.05, 0.1),
        model="TEST",
        period="held-out",
    )
    assert len(rows) == 8
    assert {row["stage"] for row in rows} == {"raw", "corrected"}


def test_projection_rows_expose_change_signal_difference() -> None:
    baseline = _datasets(1.0, 0.0)
    future_raw = {
        variable: data * (1.2 if variable == "pr" else 1.0) + (0 if variable == "pr" else 2)
        for variable, data in baseline.items()
    }
    future_corrected = {
        variable: data * (1.2 if variable == "pr" else 1.0) + (0 if variable == "pr" else 2)
        for variable, data in baseline.items()
    }
    rows = projection_change_rows(
        baseline,
        baseline,
        future_raw,
        future_corrected,
        model="TEST",
        scenario="ssp245",
        period="near-term",
    )
    assert len(rows) == 4 * 9
    assert max(abs(row["signal_difference"]) for row in rows) < 1e-9


def test_etccdi_style_totals_use_one_mm_wet_day_threshold() -> None:
    data = _datasets(1.0, 0.0)
    data["pr"] = xr.full_like(data["pr"], 0.5)
    indices = annual_extremes(data)
    assert (indices["PRCPTOT"] == 0).all()
    assert (indices["CDD"] >= 365).all()
