"""Publication-oriented figures for QDM evaluation and projections."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm.config import FigureConfig
from cloud_qdm.diagnostics import (
    PLOT_UNITS,
    QUANTILES,
    VARIABLE_LABELS,
    annual_extremes,
    climatology_map,
    correlation_matrix,
    evaluation_rows,
    monthly_cycle,
    projection_change_rows,
    quantile_values,
    spatial_mean,
    wet_day_cycle,
)

REFERENCE_COLOR = "#222222"
RAW_COLOR = "#D55E00"
CORRECTED_COLOR = "#0072B2"
STAGE_COLORS = {"reference": REFERENCE_COLOR, "raw": RAW_COLOR, "corrected": CORRECTED_COLOR}
MARKERS = {"tas": "o", "tasmax": "^", "tasmin": "v", "pr": "s"}


class FigureError(RuntimeError):
    """Raised when publication diagnostics cannot be generated safely."""


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise FigureError("Install matplotlib to generate paper figures.") from exc
    return plt


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, base: Path, settings: FigureConfig) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in settings.formats:
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=settings.dpi, bbox_inches="tight")
        paths.append(path)
    _pyplot().close(fig)
    return paths


def _finite_limit(*values: np.ndarray, floor: float = 1.0) -> float:
    valid = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    valid = valid[np.isfinite(valid)]
    return max(float(np.nanmax(np.abs(valid))) if valid.size else floor, floor)


def _line_legend(ax) -> None:
    ax.legend(loc="best", ncols=3, fontsize=8)


def _draw_taylor_axis(
    ax,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    label_models: bool,
    show_explanation: bool = True,
) -> None:
    correlations = np.array([row["correlation"] for row in rows], dtype=float)
    ratios = np.array([row["std_ratio"] for row in rows], dtype=float)
    valid = np.isfinite(correlations) & np.isfinite(ratios)
    radial_max = max(1.5, float(np.nanmax(ratios[valid])) * 1.15 if valid.any() else 1.5)

    theta = np.linspace(0, np.pi, 181)
    radius = np.linspace(0, radial_max, 160)
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    centered_rmse = np.sqrt(1 + radius_grid**2 - 2 * radius_grid * np.cos(theta_grid))
    contours = ax.contour(theta_grid, radius_grid, centered_rmse, colors="0.72", linewidths=0.6)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f")

    correlation_ticks = np.array([-1, -0.5, 0, 0.5, 0.7, 0.9, 1.0])
    ax.set_xticks(np.arccos(correlation_ticks))
    ax.set_xticklabels([f"{value:g}" for value in correlation_ticks])
    ax.set_ylim(0, radial_max)
    ax.set_title(title, pad=18, fontweight="bold")
    ax.set_xlabel("Correlation", labelpad=12)
    if show_explanation:
        ax.text(
            0.02,
            0.02,
            "Radius: normalized SD\nContours: centered RMSE",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
        )
    ax.plot(0, 1, marker="*", markersize=10, color=REFERENCE_COLOR, label="Reference")

    seen: set[str] = set()
    for row in rows:
        correlation = float(row["correlation"])
        ratio = float(row["std_ratio"])
        if not np.isfinite(correlation) or not np.isfinite(ratio):
            continue
        stage = str(row["stage"])
        variable = str(row["variable"])
        label = stage.title() if stage not in seen else None
        seen.add(stage)
        ax.scatter(
            np.arccos(np.clip(correlation, -1, 1)),
            ratio,
            color=STAGE_COLORS[stage],
            marker=MARKERS[variable],
            s=34,
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=3,
        )
        if label_models and stage == "corrected":
            ax.annotate(
                str(row["model"]),
                (np.arccos(np.clip(correlation, -1, 1)), ratio),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)


def _plot_taylor(
    rows: Sequence[dict[str, Any]],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
    facet_variables: bool,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    if facet_variables:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), subplot_kw={"projection": "polar"})
        for ax, variable in zip(axes.flat, VARIABLE_LABELS, strict=True):
            selected = [row for row in rows if row["variable"] == variable]
            _draw_taylor_axis(
                ax,
                selected,
                title=f"{VARIABLE_LABELS[variable]} ({PLOT_UNITS[variable]})",
                label_models=True,
                show_explanation=False,
            )
        fig.suptitle(title, y=0.99, fontsize=13, fontweight="bold")
        fig.text(
            0.5,
            0.955,
            "Radius: normalized standard deviation; contours: normalized centered RMSE",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(hspace=0.42, wspace=0.24, top=0.86)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), subplot_kw={"projection": "polar"})
        _draw_taylor_axis(ax, rows, title=title, label_models=False)
        stage_legend = ax.get_legend()
        if stage_legend:
            ax.add_artist(stage_legend)
        variable_handles = [
            plt.Line2D(
                [],
                [],
                color="0.35",
                marker=MARKERS[variable],
                linestyle="None",
                label=variable,
            )
            for variable in VARIABLE_LABELS
        ]
        ax.legend(handles=variable_handles, loc="lower right", fontsize=8)
    return _save(fig, base, settings)


def _plot_skill_improvement(
    rows: Sequence[dict[str, Any]], base: Path, settings: FigureConfig, *, title: str
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    metrics = {
        "|bias| / reference SD": "normalized_bias",
        "RMSE / reference SD": "normalized_rmse",
        "Centered RMSE": "centered_rmse",
        "|SD ratio - 1|": "std_ratio",
        "1 - correlation": "correlation",
    }
    matrix = np.full((len(VARIABLE_LABELS), len(metrics)), np.nan)
    for row_index, variable in enumerate(VARIABLE_LABELS):
        by_stage = {row["stage"]: row for row in rows if row["variable"] == variable}
        for column, key in enumerate(metrics.values()):
            raw_value = float(by_stage["raw"][key])
            corrected_value = float(by_stage["corrected"][key])
            if key in {"normalized_bias"}:
                raw_error, corrected_error = abs(raw_value), abs(corrected_value)
            elif key == "std_ratio":
                raw_error, corrected_error = abs(raw_value - 1), abs(corrected_value - 1)
            elif key == "correlation":
                raw_error, corrected_error = 1 - raw_value, 1 - corrected_value
            else:
                raw_error, corrected_error = raw_value, corrected_value
            matrix[row_index, column] = (
                100 * (raw_error - corrected_error) / raw_error if raw_error else np.nan
            )

    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    image = ax.imshow(np.clip(matrix, -100, 100), cmap="RdYlBu", vmin=-100, vmax=100)
    ax.set_xticks(range(len(metrics)), labels=list(metrics), rotation=25, ha="right")
    ax.set_yticks(range(len(VARIABLE_LABELS)), labels=list(VARIABLE_LABELS))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if np.isfinite(value) and abs(value) >= 55 else "black"
            ax.text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.0f}%",
                ha="center",
                va="center",
                color=color,
            )
    ax.set_title(title, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Error reduction after QDM (%)")
    fig.tight_layout()
    return _save(fig, base, settings)


def _plot_seasonal_cycle(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True)
    months = np.arange(1, 13)
    for ax, variable in zip(axes.flat, VARIABLE_LABELS, strict=True):
        for stage, source in (
            ("reference", references),
            ("raw", raw),
            ("corrected", corrected),
        ):
            ax.plot(
                months,
                monthly_cycle(source[variable], variable),
                color=STAGE_COLORS[stage],
                linewidth=1.8,
                linestyle="--" if stage == "raw" else "-",
                label=stage.title(),
            )
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_ylabel(PLOT_UNITS[variable])
        ax.set_xticks(months)
        ax.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("Calendar month")
    axes[-1, 1].set_xlabel("Calendar month")
    _line_legend(axes[0, 0])
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _plot_distribution(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for column, variable in enumerate(VARIABLE_LABELS):
        reference = quantile_values(references[variable], variable)
        raw_values = quantile_values(raw[variable], variable)
        corrected_values = quantile_values(corrected[variable], variable)
        lower = np.nanmin([reference, raw_values, corrected_values])
        upper = np.nanmax([reference, raw_values, corrected_values])

        qq_axis = axes[0, column]
        qq_axis.plot([lower, upper], [lower, upper], color="0.55", linewidth=1, label="1:1")
        qq_axis.plot(reference, raw_values, "o--", color=RAW_COLOR, markersize=3.5, label="Raw")
        qq_axis.plot(
            reference,
            corrected_values,
            "o-",
            color=CORRECTED_COLOR,
            markersize=3.5,
            label="Corrected",
        )
        qq_axis.set_title(VARIABLE_LABELS[variable])
        qq_axis.set_xlabel(f"Reference quantile ({PLOT_UNITS[variable]})")
        qq_axis.set_ylabel(f"Model quantile ({PLOT_UNITS[variable]})")
        qq_axis.grid(alpha=0.2)

        bias_axis = axes[1, column]
        if variable == "pr":
            raw_bias = 100 * np.divide(
                raw_values - reference,
                reference,
                out=np.full_like(reference, np.nan),
                where=reference != 0,
            )
            corrected_bias = 100 * np.divide(
                corrected_values - reference,
                reference,
                out=np.full_like(reference, np.nan),
                where=reference != 0,
            )
            ylabel = "Quantile bias (%)"
        else:
            raw_bias = raw_values - reference
            corrected_bias = corrected_values - reference
            ylabel = "Quantile bias (°C)"
        bias_axis.axhline(0, color="0.55", linewidth=1)
        bias_axis.plot(QUANTILES, raw_bias, "o--", color=RAW_COLOR, markersize=3.5, label="Raw")
        bias_axis.plot(
            QUANTILES,
            corrected_bias,
            "o-",
            color=CORRECTED_COLOR,
            markersize=3.5,
            label="Corrected",
        )
        bias_axis.set_xlabel("Non-exceedance probability")
        bias_axis.set_ylabel(ylabel)
        bias_axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _draw_map(ax, data: xr.DataArray, *, title: str, cmap: str, limit: float | None = None):
    kwargs = {"cmap": cmap, "shading": "auto"}
    if limit is not None:
        kwargs.update({"vmin": -limit, "vmax": limit})
    mesh = ax.pcolormesh(data["lon"], data["lat"], data, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    return mesh


def _plot_bias_map(
    reference: xr.DataArray,
    raw: xr.DataArray,
    corrected: xr.DataArray,
    variable: str,
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    reference_map = climatology_map(reference, variable)
    raw_map = climatology_map(raw, variable)
    corrected_map = climatology_map(corrected, variable)
    if variable == "pr":
        valid = np.abs(reference_map) >= 0.1
        raw_bias = 100 * (raw_map - reference_map) / reference_map.where(valid)
        corrected_bias = 100 * (corrected_map - reference_map) / reference_map.where(valid)
        bias_units = "%"
    else:
        raw_bias = raw_map - reference_map
        corrected_bias = corrected_map - reference_map
        bias_units = "°C"
    improvement = np.abs(raw_bias) - np.abs(corrected_bias)
    bias_limit = _finite_limit(raw_bias.values, corrected_bias.values)
    improvement_limit = _finite_limit(improvement.values)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    reference_mesh = _draw_map(
        axes[0],
        reference_map,
        title=f"Reference mean ({PLOT_UNITS[variable]})",
        cmap="viridis",
    )
    raw_mesh = _draw_map(
        axes[1], raw_bias, title=f"Raw bias ({bias_units})", cmap="RdBu_r", limit=bias_limit
    )
    _draw_map(
        axes[2],
        corrected_bias,
        title=f"Corrected bias ({bias_units})",
        cmap="RdBu_r",
        limit=bias_limit,
    )
    improvement_mesh = _draw_map(
        axes[3],
        improvement,
        title=f"Absolute-bias reduction ({bias_units})",
        cmap="BrBG",
        limit=improvement_limit,
    )
    fig.colorbar(reference_mesh, ax=axes[0], shrink=0.77)
    fig.colorbar(raw_mesh, ax=axes[1:3], shrink=0.77)
    fig.colorbar(improvement_mesh, ax=axes[3], shrink=0.77)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save(fig, base, settings)


def _plot_wet_days(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    threshold: float,
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    months = np.arange(1, 13)
    for stage, source in (
        ("reference", references),
        ("raw", raw),
        ("corrected", corrected),
    ):
        frequency, intensity = wet_day_cycle(source["pr"], threshold)
        style = "--" if stage == "raw" else "-"
        axes[0].plot(months, frequency, style, color=STAGE_COLORS[stage], label=stage.title())
        axes[1].plot(months, intensity, style, color=STAGE_COLORS[stage], label=stage.title())
    axes[0].set_title(f"Wet-day frequency (pr ≥ {threshold:g} mm d⁻¹)")
    axes[0].set_ylabel("Days (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Wet-day mean intensity")
    axes[1].set_ylabel("mm d⁻¹")
    for ax in axes:
        ax.set_xlabel("Calendar month")
        ax.set_xticks(months)
        ax.grid(alpha=0.2)
    _line_legend(axes[0])
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _plot_extremes(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    frames = {
        "reference": annual_extremes(references),
        "raw": annual_extremes(raw),
        "corrected": annual_extremes(corrected),
    }
    labels = {
        "PRCPTOT": "Annual precipitation (mm)",
        "Rx1day": "Rx1day (mm)",
        "Rx5day": "Rx5day (mm)",
        "CDD": "Maximum dry spell (days)",
        "TXx": "TXx (°C)",
        "TNn": "TNn (°C)",
        "DTR": "Mean daily temperature range (°C)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for ax, (index, ylabel) in zip(axes.flat, labels.items(), strict=False):
        for stage, frame in frames.items():
            ax.plot(
                frame.index,
                frame[index],
                color=STAGE_COLORS[stage],
                linestyle="--" if stage == "raw" else "-",
                marker="o",
                markersize=2.5,
                label=stage.title(),
            )
        ax.set_title(index)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, legend_labels, loc="center", fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _plot_dependence(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    image = None
    for ax, (stage, source) in zip(
        axes,
        (("Reference", references), ("Raw model", raw), ("Corrected", corrected)),
        strict=True,
    ):
        matrix = correlation_matrix(source)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), labels=list(VARIABLE_LABELS), rotation=35, ha="right")
        ax.set_yticks(range(4), labels=list(VARIABLE_LABELS))
        ax.set_title(stage)
        for row in range(4):
            for column in range(4):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[row, column]) >= 0.55 else "black",
                )
    fig.colorbar(image, ax=axes, shrink=0.78, label="Pearson correlation")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save(fig, base, settings)


def make_evaluation_figures(
    references: Mapping[str, xr.DataArray],
    raw: Mapping[str, xr.DataArray],
    corrected: Mapping[str, xr.DataArray],
    *,
    model: str,
    period: str,
    output_dir: Path,
    settings: FigureConfig,
    wet_day_threshold: float,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Generate the independent historical evaluation figure suite."""
    rows = evaluation_rows(references, raw, corrected, model=model, period=period)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "skill-metrics.csv", index=False)
    paths: list[Path] = []
    paths += _plot_taylor(
        rows,
        output_dir / "taylor-diagram",
        settings,
        title=f"Independent evaluation: {model} ({period})",
        facet_variables=False,
    )
    paths += _plot_skill_improvement(
        rows,
        output_dir / "skill-improvement",
        settings,
        title=f"QDM error reduction: {model} ({period})",
    )
    paths += _plot_seasonal_cycle(
        references,
        raw,
        corrected,
        output_dir / "seasonal-cycle",
        settings,
        title=f"Monthly climatology: {model} ({period})",
    )
    paths += _plot_distribution(
        references,
        raw,
        corrected,
        output_dir / "distribution-and-quantile-bias",
        settings,
        title=f"Distribution diagnostics: {model} ({period})",
    )
    for variable in VARIABLE_LABELS:
        paths += _plot_bias_map(
            references[variable],
            raw[variable],
            corrected[variable],
            variable,
            output_dir / f"spatial-bias-{variable}",
            settings,
            title=f"Spatial bias: {VARIABLE_LABELS[variable]} — {model} ({period})",
        )
    paths += _plot_wet_days(
        references,
        raw,
        corrected,
        wet_day_threshold,
        output_dir / "wet-day-diagnostics",
        settings,
        title=f"Precipitation occurrence and intensity: {model} ({period})",
    )
    paths += _plot_extremes(
        references,
        raw,
        corrected,
        output_dir / "annual-extremes",
        settings,
        title=f"Annual regional extremes: {model} ({period})",
    )
    paths += _plot_dependence(
        references,
        raw,
        corrected,
        output_dir / "intervariable-dependence",
        settings,
        title=f"Inter-variable dependence diagnostic: {model} ({period})",
    )
    return paths, rows


def _change_map(historical: xr.DataArray, future: xr.DataArray, variable: str) -> xr.DataArray:
    historical_map = climatology_map(historical, variable)
    future_map = climatology_map(future, variable)
    if variable == "pr":
        valid = np.abs(historical_map) >= 0.1
        return 100 * (future_map - historical_map) / historical_map.where(valid)
    return future_map - historical_map


def _plot_change_maps(
    baseline_raw: Mapping[str, xr.DataArray],
    baseline_corrected: Mapping[str, xr.DataArray],
    future_raw: Mapping[str, xr.DataArray],
    future_corrected: Mapping[str, xr.DataArray],
    variable: str,
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    raw_change = _change_map(baseline_raw[variable], future_raw[variable], variable)
    corrected_change = _change_map(
        baseline_corrected[variable], future_corrected[variable], variable
    )
    difference = corrected_change - raw_change
    limit = _finite_limit(raw_change.values, corrected_change.values)
    difference_limit = _finite_limit(difference.values)
    units = "%" if variable == "pr" else "°C"

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7), constrained_layout=True)
    raw_mesh = _draw_map(
        axes[0], raw_change, title=f"Raw change ({units})", cmap="RdBu_r", limit=limit
    )
    _draw_map(
        axes[1],
        corrected_change,
        title=f"Corrected change ({units})",
        cmap="RdBu_r",
        limit=limit,
    )
    difference_mesh = _draw_map(
        axes[2],
        difference,
        title=f"QDM minus raw signal ({units})",
        cmap="PuOr_r",
        limit=difference_limit,
    )
    fig.colorbar(raw_mesh, ax=axes[:2], shrink=0.77)
    fig.colorbar(difference_mesh, ax=axes[2], shrink=0.77)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save(fig, base, settings)


def _plot_change_signal(
    rows: Sequence[dict[str, Any]], base: Path, settings: FigureConfig, *, title: str
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, variable in zip(axes.flat, VARIABLE_LABELS, strict=True):
        selected = frame[frame["variable"] == variable]
        ax.axhline(0, color="0.55", linewidth=1)
        ax.plot(
            selected["quantile"],
            selected["raw_change"],
            "o--",
            color=RAW_COLOR,
            markersize=3.5,
            label="Raw model",
        )
        ax.plot(
            selected["quantile"],
            selected["corrected_change"],
            "o-",
            color=CORRECTED_COLOR,
            markersize=3.5,
            label="After QDM",
        )
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_ylabel(f"Future change ({selected['units'].iloc[0]})")
        ax.set_xlabel("Non-exceedance probability")
        ax.grid(alpha=0.2)
    _line_legend(axes[0, 0])
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _annual_mean(data: xr.DataArray, variable: str) -> tuple[np.ndarray, np.ndarray]:
    values = spatial_mean(data - 273.15 if variable != "pr" else data)
    reducer = (
        values.groupby("time.year").sum if variable == "pr" else values.groupby("time.year").mean
    )
    annual = reducer(skipna=True).compute()
    return np.asarray(annual["year"].values), np.asarray(annual.values, dtype=float)


def _plot_projection_series(
    future_raw: Mapping[str, xr.DataArray],
    future_corrected: Mapping[str, xr.DataArray],
    base: Path,
    settings: FigureConfig,
    *,
    title: str,
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, variable in zip(axes.flat, VARIABLE_LABELS, strict=True):
        raw_years, raw_values = _annual_mean(future_raw[variable], variable)
        corrected_years, corrected_values = _annual_mean(future_corrected[variable], variable)
        ax.plot(raw_years, raw_values, "--", color=RAW_COLOR, linewidth=1.3, label="Raw model")
        ax.plot(
            corrected_years,
            corrected_values,
            "-",
            color=CORRECTED_COLOR,
            linewidth=1.5,
            label="After QDM",
        )
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_ylabel("mm yr⁻¹" if variable == "pr" else "°C")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)
    _line_legend(axes[0, 0])
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def make_projection_figures(
    baseline_raw: Mapping[str, xr.DataArray],
    baseline_corrected: Mapping[str, xr.DataArray],
    future_raw: Mapping[str, xr.DataArray],
    future_corrected: Mapping[str, xr.DataArray],
    *,
    model: str,
    scenario: str,
    period: str,
    output_dir: Path,
    settings: FigureConfig,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Generate future change, signal-preservation, and time-series figures."""
    rows = projection_change_rows(
        baseline_raw,
        baseline_corrected,
        future_raw,
        future_corrected,
        model=model,
        scenario=scenario,
        period=period,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "quantile-change.csv", index=False)
    paths = _plot_change_signal(
        rows,
        output_dir / "quantile-change-signal",
        settings,
        title=f"Quantile change preservation: {model}, {scenario}, {period}",
    )
    paths += _plot_projection_series(
        future_raw,
        future_corrected,
        output_dir / "annual-projection-series",
        settings,
        title=f"Annual regional projections: {model}, {scenario}, {period}",
    )
    for variable in VARIABLE_LABELS:
        paths += _plot_change_maps(
            baseline_raw,
            baseline_corrected,
            future_raw,
            future_corrected,
            variable,
            output_dir / f"change-map-{variable}",
            settings,
            title=f"Projected {VARIABLE_LABELS[variable].lower()} change: {model}, {scenario}, {period}",
        )
    return paths, rows


def _plot_ensemble_skill(
    rows: Sequence[dict[str, Any]], base: Path, settings: FigureConfig
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    frame = pd.DataFrame(rows)
    models = list(dict.fromkeys(frame["model"]))
    matrix = np.full((len(models), len(VARIABLE_LABELS)), np.nan)
    for row_index, model in enumerate(models):
        for column, variable in enumerate(VARIABLE_LABELS):
            selected = frame[(frame["model"] == model) & (frame["variable"] == variable)]
            raw = float(selected[selected["stage"] == "raw"]["normalized_rmse"].iloc[0])
            corrected = float(selected[selected["stage"] == "corrected"]["normalized_rmse"].iloc[0])
            matrix[row_index, column] = 100 * (raw - corrected) / raw if raw else np.nan
    fig_height = max(3.5, 0.35 * len(models) + 1.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    image = ax.imshow(np.clip(matrix, -100, 100), cmap="RdYlBu", vmin=-100, vmax=100, aspect="auto")
    ax.set_xticks(range(4), labels=list(VARIABLE_LABELS))
    ax.set_yticks(range(len(models)), labels=models)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[row, column]) >= 55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Normalized RMSE reduction (%)")
    ax.set_title("Independent-evaluation skill improvement", fontweight="bold")
    fig.tight_layout()
    return _save(fig, base, settings)


def _plot_ensemble_change(
    rows: Sequence[dict[str, Any]], base: Path, settings: FigureConfig
) -> list[Path]:
    plt = _pyplot()
    _style(plt)
    frame = pd.DataFrame(rows)
    frame = frame[np.isclose(frame["quantile"], 0.5)]
    groups = list(dict.fromkeys(zip(frame["scenario"], frame["period"], strict=True)))
    labels = [f"{scenario}\n{period}" for scenario, period in groups]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(groups) * 1.6), 7), sharex=True)
    for ax, variable in zip(axes.flat, VARIABLE_LABELS, strict=True):
        selected = frame[frame["variable"] == variable]
        raw_values = [
            selected[(selected["scenario"] == scenario) & (selected["period"] == period)][
                "raw_change"
            ].to_numpy()
            for scenario, period in groups
        ]
        corrected_values = [
            selected[(selected["scenario"] == scenario) & (selected["period"] == period)][
                "corrected_change"
            ].to_numpy()
            for scenario, period in groups
        ]
        positions = np.arange(len(groups))
        raw_box = ax.boxplot(raw_values, positions=positions - 0.17, widths=0.28, patch_artist=True)
        corrected_box = ax.boxplot(
            corrected_values, positions=positions + 0.17, widths=0.28, patch_artist=True
        )
        for patch in raw_box["boxes"]:
            patch.set_facecolor(RAW_COLOR)
            patch.set_alpha(0.65)
        for patch in corrected_box["boxes"]:
            patch.set_facecolor(CORRECTED_COLOR)
            patch.set_alpha(0.65)
        ax.axhline(0, color="0.55", linewidth=0.8)
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_ylabel(f"Median-quantile change ({selected['units'].iloc[0]})")
        ax.set_xticks(positions, labels=labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    handles = [
        plt.Line2D([], [], color=RAW_COLOR, linewidth=7, alpha=0.65, label="Raw model"),
        plt.Line2D([], [], color=CORRECTED_COLOR, linewidth=7, alpha=0.65, label="After QDM"),
    ]
    axes[0, 0].legend(handles=handles)
    fig.suptitle(
        "Multi-model projected change by scenario and window", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, base, settings)


def make_ensemble_figures(
    evaluation: Sequence[dict[str, Any]],
    projections: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    settings: FigureConfig,
) -> list[Path]:
    """Generate cross-model core figures and machine-readable metric tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation).to_csv(output_dir / "evaluation-metrics.csv", index=False)
    pd.DataFrame(projections).to_csv(output_dir / "projection-change.csv", index=False)
    paths = _plot_taylor(
        evaluation,
        output_dir / "evaluation-taylor-by-variable",
        settings,
        title="Independent historical evaluation across models",
        facet_variables=True,
    )
    paths += _plot_ensemble_skill(evaluation, output_dir / "evaluation-skill-improvement", settings)
    if projections:
        paths += _plot_ensemble_change(
            projections, output_dir / "projection-ensemble-change", settings
        )
    manifest = {
        "note": "Figures are diagnostics, not automatic evidence of fitness for impact use.",
        "figures": [str(path) for path in paths],
    }
    (output_dir / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths
