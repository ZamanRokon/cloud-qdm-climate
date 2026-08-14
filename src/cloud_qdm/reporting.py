"""NetCDF output, summary statistics, logging, and run provenance."""

from __future__ import annotations

import json
import logging
import platform
import sys
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


def save_netcdf(data: xr.DataArray, path: Path, attributes: dict[str, Any]) -> Path:
    """Materialize a corrected DataArray as a compressed NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = data.where(np.isfinite(data)).astype(np.float32).copy()
    output.attrs.pop("_FillValue", None)
    output.attrs.pop("missing_value", None)
    output.attrs.update({key: str(value) for key, value in attributes.items()})
    output.attrs["created_utc"] = datetime.now(UTC).isoformat()
    output.attrs["software"] = "cloud-qdm-climate"
    output.attrs["software_version"] = __version__
    encoding = {
        output.name or "climate": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    output.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


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
