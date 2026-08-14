"""NetCDF output, summary statistics, logging, and run provenance."""

from __future__ import annotations

import json
import logging
import platform
import sys
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from cloud_qdm import __version__


def configure_logging(run_dir: Path) -> logging.Logger:
    """Create an idempotent console and UTF-8 file logger."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cloud_qdm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _netcdf_payload(
    data: xr.DataArray,
    attributes: dict[str, Any],
    compression_level: int,
) -> tuple[xr.DataArray, dict[str, dict[str, Any]]]:
    """Build a finite float32 payload and its NetCDF encoding."""
    output = data.where(np.isfinite(data)).astype(np.float32).copy()
    output.attrs.pop("_FillValue", None)
    output.attrs.pop("missing_value", None)
    output.attrs.update({key: str(value) for key, value in attributes.items()})
    output.attrs["created_utc"] = datetime.now(UTC).isoformat()
    output.attrs["software"] = "cloud-qdm-climate"
    output.attrs["software_version"] = __version__
    encoding = {
        output.name or "climate": {
            "zlib": compression_level > 0,
            "complevel": compression_level,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    return output, encoding


def save_netcdf(
    data: xr.DataArray,
    path: Path,
    attributes: dict[str, Any],
    *,
    compression_level: int = 1,
) -> Path:
    """Materialize a corrected DataArray as a compressed NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output, encoding = _netcdf_payload(data, attributes, compression_level)
    output.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def save_netcdf_batch(
    items: Iterable[tuple[xr.DataArray, Path, dict[str, Any]]],
    *,
    compression_level: int = 1,
    staging_dir: Path | None = None,
) -> list[Path]:
    """Evaluate shared arrays once in a staging dataset, then split the files."""
    prepared: list[tuple[xr.DataArray, Path, dict[str, dict[str, Any]]]] = []
    paths: list[Path] = []
    for data, path, attributes in items:
        path.parent.mkdir(parents=True, exist_ok=True)
        output, encoding = _netcdf_payload(data, attributes, compression_level)
        prepared.append((output, path, encoding))
        paths.append(path)
    if not prepared:
        return paths
    if len(prepared) == 1:
        output, path, encoding = prepared[0]
        output.to_netcdf(path, engine="netcdf4", encoding=encoding)
        return paths

    variable_names = [output.name for output, _, _ in prepared]
    if any(name is None for name in variable_names) or len(set(variable_names)) != len(prepared):
        raise ValueError("Batched NetCDF variables must have unique names.")
    try:
        xr.align(*(output for output, _, _ in prepared), join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("Batched NetCDF variables must use exactly the same coordinates.") from exc

    stage_parent = staging_dir or paths[0].parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".qdm-stage-", suffix=".nc", dir=stage_parent, delete=False
    ) as handle:
        stage_path = Path(handle.name)
    stage_encoding = {
        str(output.name): {
            "zlib": False,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
        for output, _, _ in prepared
    }
    try:
        dataset = xr.merge(
            [output.to_dataset(name=str(output.name)) for output, _, _ in prepared],
            join="exact",
        )
        dataset.to_netcdf(stage_path, engine="netcdf4", encoding=stage_encoding)
        with xr.open_dataset(stage_path, chunks="auto") as staged:
            for output, path, encoding in prepared:
                staged[str(output.name)].to_netcdf(
                    path,
                    engine="netcdf4",
                    encoding=encoding,
                )
    finally:
        stage_path.unlink(missing_ok=True)
    return paths


def summarize(
    data: xr.DataArray,
    *,
    model: str,
    scenario: str,
    period: str,
    variable: str,
    output_path: Path,
) -> dict[str, Any]:
    """Compute compact whole-cube quality-control statistics."""
    finite = np.isfinite(data)
    clean = data.where(finite)
    stats = xr.Dataset(
        {
            "mean": clean.mean(skipna=True),
            "std": clean.std(skipna=True),
            "minimum": clean.min(skipna=True),
            "maximum": clean.max(skipna=True),
            "quantiles": clean.quantile([0.05, 0.5, 0.95], dim=list(data.dims), skipna=True),
            "finite_count": finite.sum(),
            "zero_count": (clean == 0).sum(),
        }
    ).compute()
    quantiles = stats["quantiles"]
    finite_count = int(stats["finite_count"].item())
    total_count = int(data.size)
    return {
        "model": model,
        "scenario": scenario,
        "period": period,
        "variable": variable,
        "units": data.attrs.get("units", ""),
        "mean": float(stats["mean"].item()),
        "std": float(stats["std"].item()),
        "min": float(stats["minimum"].item()),
        "p05": float(quantiles.sel(quantile=0.05).item()),
        "p50": float(quantiles.sel(quantile=0.5).item()),
        "p95": float(quantiles.sel(quantile=0.95).item()),
        "max": float(stats["maximum"].item()),
        "total_values": total_count,
        "finite_values": finite_count,
        "non_finite_values": total_count - finite_count,
        "finite_fraction": finite_count / total_count if total_count else 0.0,
        "zero_values": int(stats["zero_count"].item()),
        "path": str(output_path),
    }


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def base_manifest() -> dict[str, Any]:
    dependencies = {}
    for package in ("dask", "earthengine-api", "numpy", "xarray", "xee", "xsdba"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
    return {
        "software": "cloud-qdm-climate",
        "software_version": __version__,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "status": "running",
        "models": {},
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
