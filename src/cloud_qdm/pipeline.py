"""End-to-end model-by-model QDM pipeline."""

from __future__ import annotations

import gc
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from cloud_qdm.config import Period, RunConfig
from cloud_qdm.coordinates import (
    align_calibration_time,
    enforce_temperature_ordering,
    regrid_to_reference,
)
from cloud_qdm.diagnostics import (
    evaluation_rows as build_evaluation_rows,
)
from cloud_qdm.diagnostics import (
    projection_change_rows,
)
from cloud_qdm.figures import (
    make_ensemble_figures,
    make_evaluation_figures,
    make_projection_figures,
)
from cloud_qdm.qdm import apply_qdm, save_adjustment, train_qdm
from cloud_qdm.reporting import (
    base_manifest,
    configure_logging,
    save_netcdf,
    save_netcdf_batch,
    summarize,
    write_manifest,
    write_summary,
)
from cloud_qdm.sources import (
    VARIABLES,
    fetch_chirps_reference,
    fetch_era5_references,
    fetch_gddp,
    fetch_gddp_variables,
    initialize_earth_engine,
    open_mswep,
)

DataByVariable = dict[str, Any]
EvaluationData = tuple[DataByVariable, DataByVariable, DataByVariable]
MODEL_FETCH_PADDING_DEGREES = 0.5
FUTURE_OUTPUT_LABEL = "2015-2100"


def _load_references(config: RunConfig, logger) -> DataByVariable:
    logger.info("Loading ERA5-Land temperature references in one batched request")
    references = fetch_era5_references(
        collection_id=config.earth_engine.era5_collection,
        period=config.calibration,
        bounds=config.aoi,
        chunk_years=config.processing.earth_engine_chunk_years,
    )
    if config.precipitation_reference.mode == "chirps":
        logger.info("Loading CHIRPS precipitation reference")
        references["pr"] = fetch_chirps_reference(
            collection_id=config.precipitation_reference.chirps_collection,
            period=config.calibration,
            bounds=config.aoi,
            chunk_years=config.processing.earth_engine_chunk_years,
        )
    else:
        logger.info("Loading user-supplied MSWEP precipitation reference")
        references["pr"] = open_mswep(
            config.precipitation_reference.mswep,
            period=config.calibration,
            bounds=config.aoi,
            chunks={
                "lat": config.processing.latitude_chunk,
                "lon": config.processing.longitude_chunk,
            },
        )
        replaced = int(references["pr"].attrs.get("non_finite_values_replaced", 0))
        if replaced:
            logger.warning(
                "Replaced %s non-finite MSWEP values with zero under the configured policy",
                f"{replaced:,}",
            )
    return references


def _reference_source(config: RunConfig, variable: str) -> str:
    if variable != "pr":
        return config.earth_engine.era5_collection
    if config.precipitation_reference.mode == "chirps":
        return config.precipitation_reference.chirps_collection
    return "MSWEP (user supplied; local path withheld from NetCDF metadata)"


def _fetch_model(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    period,
    variable: str,
):
    return fetch_gddp(
        collection_id=config.earth_engine.gddp_collection,
        variable=variable,
        model=model,
        scenario=scenario,
        period=period,
        bounds=config.aoi.padded(MODEL_FETCH_PADDING_DEGREES),
        chunk_years=config.processing.earth_engine_chunk_years,
        grid_label=config.grid_labels.get(model),
    )


def _fetch_model_variables(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    period,
) -> DataByVariable:
    return fetch_gddp_variables(
        collection_id=config.earth_engine.gddp_collection,
        model=model,
        scenario=scenario,
        period=period,
        bounds=config.aoi.padded(MODEL_FETCH_PADDING_DEGREES),
        chunk_years=config.processing.earth_engine_chunk_years,
        grid_label=config.grid_labels.get(model),
    )


def _correct_period(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    period,
    references: DataByVariable,
    adjustments: DataByVariable,
    preloaded: DataByVariable | None = None,
) -> tuple[DataByVariable, DataByVariable]:
    corrected = {}
    model_inputs = {}
    raw_inputs = (
        None
        if preloaded is not None
        else _fetch_model_variables(
            config,
            model=model,
            scenario=scenario,
            period=period,
        )
    )
    for variable in VARIABLES:
        if preloaded is not None:
            model_on_reference = preloaded[variable]
        else:
            raw = raw_inputs[variable]
            model_on_reference = regrid_to_reference(raw, references[variable])
        model_inputs[variable] = model_on_reference
        corrected[variable] = apply_qdm(
            adjustments[variable],
            model_on_reference,
            variable=variable,
            config=config.qdm,
        )
    ordered = enforce_temperature_ordering(
        corrected["tas"], corrected["tasmax"], corrected["tasmin"]
    )
    corrected.update(ordered)
    return corrected, model_inputs


def _save_period(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    period,
    corrected: DataByVariable,
    summary_rows: list[dict[str, Any]],
    output_root: Path | None = None,
    record_summary: bool = True,
    compression_level: int | None = None,
) -> list[str]:
    output_paths: list[str] = []
    provenance = {
        "model": model,
        "scenario": scenario,
        "period_start": period.start.isoformat(),
        "period_end": period.end.isoformat(),
        "calibration_start": config.calibration.start.isoformat(),
        "calibration_end": config.calibration.end.isoformat(),
        "aoi_bounds": config.aoi.as_list(),
        "precipitation_reference": config.precipitation_reference.mode,
        "model_source_collection": config.earth_engine.gddp_collection,
    }
    root = output_root or config.run_dir / "corrected"
    write_items = []
    for variable, data in corrected.items():
        output_path = root / model / scenario / period.label / f"{variable}.nc"
        write_items.append(
            (
                data,
                output_path,
                {**provenance, "reference_source": _reference_source(config, variable)},
            )
        )
    level = (
        config.processing.netcdf_compression_level
        if compression_level is None
        else compression_level
    )
    staging_dir = config.segment_dir.parent if config.processing.scratch_dir else None
    temperature_items = [item for item in write_items if item[0].name != "pr"]
    precipitation_items = [item for item in write_items if item[0].name == "pr"]
    written = save_netcdf_batch(
        temperature_items,
        compression_level=level,
        staging_dir=staging_dir,
    ) + save_netcdf_batch(
        precipitation_items,
        compression_level=level,
        staging_dir=staging_dir,
    )
    paths_by_variable = {path.stem: path for path in written}
    for variable in corrected:
        output_path = paths_by_variable[variable]
        if record_summary:
            with xr.open_dataarray(output_path, chunks="auto") as saved:
                summary_rows.append(
                    summarize(
                        saved,
                        model=model,
                        scenario=scenario,
                        period=period.label,
                        variable=variable,
                        output_path=output_path,
                    )
                )
        output_paths.append(str(output_path))
    return output_paths


def _validate_complete_future_time(data: xr.DataArray) -> None:
    """Require one unique daily value for every date from 2015 through 2100."""
    observed = pd.DatetimeIndex(pd.to_datetime(data["time"].values)).normalize()
    expected = pd.date_range("2015-01-01", "2100-12-31", freq="D")
    if observed.has_duplicates:
        raise RuntimeError("Merged future output contains duplicate dates.")
    if not observed.equals(expected):
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        raise RuntimeError(
            "Merged future output is not complete for 2015-01-01 to 2100-12-31 "
            f"(missing={len(missing):,}, unexpected={len(unexpected):,})."
        )


def _remove_segment_files(paths: list[Path], segment_root: Path) -> None:
    """Delete only successfully merged temporary files and their empty directories."""
    for path in paths:
        path.unlink()
    directories = sorted(
        {path.parent for path in paths}, key=lambda item: len(item.parts), reverse=True
    )
    for directory in directories:
        directory.rmdir()
    current = directories[-1].parent if directories else segment_root
    while current != segment_root.parent and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _merge_future_segments(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    segment_paths: dict[str, list[Path]],
    summary_rows: list[dict[str, Any]],
) -> list[str]:
    """Merge internal windows into one daily 2015-2100 NetCDF per variable."""
    final_period = Period(date(2015, 1, 1), date(2100, 12, 31), FUTURE_OUTPUT_LABEL)
    output_paths: list[str] = []
    all_segments = [path for paths in segment_paths.values() for path in paths]
    for variable in VARIABLES:
        paths = segment_paths.get(variable, [])
        if len(paths) != len(config.future_windows):
            raise RuntimeError(
                f"Expected {len(config.future_windows)} {variable} segments for "
                f"{model}/{scenario}; found {len(paths)}."
            )
        with xr.open_mfdataset(paths, combine="by_coords", chunks="auto") as dataset:
            merged = dataset[variable].sortby("time")
            _validate_complete_future_time(merged)
            output_paths.extend(
                _save_period(
                    config,
                    model=model,
                    scenario=scenario,
                    period=final_period,
                    corrected={variable: merged},
                    summary_rows=summary_rows,
                )
            )
    _remove_segment_files(all_segments, config.segment_dir)
    return output_paths


def _train_model(
    config: RunConfig,
    *,
    model: str,
    references: DataByVariable,
    logger,
) -> tuple[DataByVariable, DataByVariable, EvaluationData | None]:
    adjustments = {}
    historical_on_reference = {}
    evaluation_references = {}
    evaluation_raw = {}
    evaluation_corrected = {}
    logger.info("Fetching historical %s variables in one batched request", model)
    raw_historical = _fetch_model_variables(
        config,
        model=model,
        scenario="historical",
        period=config.calibration,
    )
    for variable in VARIABLES:
        historical = raw_historical[variable]
        historical = regrid_to_reference(historical, references[variable])
        reference, historical = align_calibration_time(
            references[variable],
            historical,
            minimum_coverage=config.processing.minimum_time_coverage,
        )
        historical_on_reference[variable] = historical

        if config.figures.enabled and config.evaluation:
            training_reference = reference.sel(
                time=slice(
                    config.evaluation.training.start.isoformat(),
                    config.evaluation.training.end.isoformat(),
                )
            )
            training_historical = historical.sel(
                time=slice(
                    config.evaluation.training.start.isoformat(),
                    config.evaluation.training.end.isoformat(),
                )
            )
            validation_reference = reference.sel(
                time=slice(
                    config.evaluation.validation.start.isoformat(),
                    config.evaluation.validation.end.isoformat(),
                )
            )
            validation_historical = historical.sel(
                time=slice(
                    config.evaluation.validation.start.isoformat(),
                    config.evaluation.validation.end.isoformat(),
                )
            )
            training_reference, training_historical = align_calibration_time(
                training_reference,
                training_historical,
                minimum_coverage=config.processing.minimum_time_coverage,
            )
            validation_reference, validation_historical = align_calibration_time(
                validation_reference,
                validation_historical,
                minimum_coverage=config.processing.minimum_time_coverage,
            )
            logger.info("Training evaluation QDM %s/%s", model, variable)
            evaluation_adjustment = train_qdm(
                training_reference,
                training_historical,
                variable=variable,
                config=config.qdm,
            )
            evaluation_references[variable] = validation_reference
            evaluation_raw[variable] = validation_historical
            evaluation_corrected[variable] = apply_qdm(
                evaluation_adjustment,
                validation_historical,
                variable=variable,
                config=config.qdm,
            )

        logger.info("Training QDM %s/%s", model, variable)
        adjustment = train_qdm(reference, historical, variable=variable, config=config.qdm)
        adjustments[variable] = adjustment
        path = config.run_dir / "adjustments" / model / f"qdm_{variable}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_adjustment(adjustment, str(path))
    evaluation = None
    if config.figures.enabled and config.evaluation:
        evaluation_corrected.update(
            enforce_temperature_ordering(
                evaluation_corrected["tas"],
                evaluation_corrected["tasmax"],
                evaluation_corrected["tasmin"],
            )
        )
        evaluation = (evaluation_references, evaluation_raw, evaluation_corrected)
    return adjustments, historical_on_reference, evaluation


def _evaluation_diagnostics(
    config: RunConfig,
    evaluation: EvaluationData,
    *,
    model: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Build core metrics and optionally render the per-model evaluation suite."""
    if config.evaluation is None:
        raise RuntimeError("Evaluation periods are required for figure diagnostics.")
    if not config.figures.model_by_model:
        return [], build_evaluation_rows(
            *evaluation,
            model=model,
            period=config.evaluation.validation.label,
        )
    return make_evaluation_figures(
        *evaluation,
        model=model,
        period=config.evaluation.validation.label,
        output_dir=config.run_dir / "figures" / "by-model" / model / "evaluation",
        settings=config.figures,
        wet_day_threshold=config.qdm.wet_day_threshold_mm,
    )


def _projection_diagnostics(
    config: RunConfig,
    baseline_raw: DataByVariable,
    baseline_corrected: DataByVariable,
    future_raw: DataByVariable,
    future_corrected: DataByVariable,
    *,
    model: str,
    scenario: str,
    period: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Build core projection metrics and optionally render per-model figures."""
    if not config.figures.model_by_model:
        return [], projection_change_rows(
            baseline_raw,
            baseline_corrected,
            future_raw,
            future_corrected,
            model=model,
            scenario=scenario,
            period=period,
        )
    return make_projection_figures(
        baseline_raw,
        baseline_corrected,
        future_raw,
        future_corrected,
        model=model,
        scenario=scenario,
        period=period,
        output_dir=(
            config.run_dir / "figures" / "by-model" / model / "projection" / scenario / period
        ),
        settings=config.figures,
    )


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    """Execute a validated run and return its manifest."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(config.run_dir)
    manifest_path = config.run_dir / "run-manifest.json"
    manifest = base_manifest()
    manifest["figures"] = []
    manifest["sources"] = {
        "model": config.earth_engine.gddp_collection,
        "temperature_reference": config.earth_engine.era5_collection,
        "precipitation_reference": (
            config.precipitation_reference.chirps_collection
            if config.precipitation_reference.mode == "chirps"
            else "MSWEP (user supplied; path recorded only in run-config.yml)"
        ),
    }
    write_manifest(manifest, manifest_path)
    (config.run_dir / "run-config.yml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    logger.info("Starting run '%s'", config.name)
    initialize_earth_engine(config.earth_engine.project_id)
    references = _load_references(config, logger)

    if config.processing.save_reference_subsets:
        for variable, data in references.items():
            save_netcdf(
                data,
                config.run_dir / "references" / f"{variable}.nc",
                {
                    "role": "calibration reference",
                    "period_start": config.calibration.start.isoformat(),
                    "period_end": config.calibration.end.isoformat(),
                },
                compression_level=config.processing.netcdf_compression_level,
            )

    summary_rows: list[dict[str, Any]] = []
    all_evaluation_rows: list[dict[str, Any]] = []
    all_projection_rows: list[dict[str, Any]] = []
    for model in config.models:
        logger.info("Processing model %s", model)
        manifest["models"][model] = {
            "status": "running",
            "outputs": [],
            "figures": [],
            "error": None,
        }
        write_manifest(manifest, manifest_path)
        try:
            model_evaluation_rows: list[dict[str, Any]] = []
            model_projection_rows: list[dict[str, Any]] = []
            adjustments, historical, evaluation = _train_model(
                config, model=model, references=references, logger=logger
            )
            if config.figures.enabled and config.evaluation and evaluation:
                if config.figures.model_by_model:
                    logger.info("Generating independent evaluation figures for %s", model)
                figure_paths, model_evaluation_rows = _evaluation_diagnostics(
                    config, evaluation, model=model
                )
                manifest["models"][model]["figures"].extend(map(str, figure_paths))

            logger.info("Correcting %s/historical/%s", model, config.calibration.label)
            baseline_corrected, _ = _correct_period(
                config,
                model=model,
                scenario="historical",
                period=config.calibration,
                references=references,
                adjustments=adjustments,
                preloaded=historical,
            )
            manifest["models"][model]["outputs"].extend(
                _save_period(
                    config,
                    model=model,
                    scenario="historical",
                    period=config.calibration,
                    corrected=baseline_corrected,
                    summary_rows=summary_rows,
                )
            )

            for scenario in config.scenarios:
                segment_paths: dict[str, list[Path]] = {variable: [] for variable in VARIABLES}
                for period in config.future_windows:
                    logger.info("Correcting %s/%s/%s", model, scenario, period.label)
                    corrected, model_inputs = _correct_period(
                        config,
                        model=model,
                        scenario=scenario,
                        period=period,
                        references=references,
                        adjustments=adjustments,
                    )
                    saved_segments = _save_period(
                        config,
                        model=model,
                        scenario=scenario,
                        period=period,
                        corrected=corrected,
                        summary_rows=summary_rows,
                        output_root=config.segment_dir,
                        record_summary=False,
                        compression_level=(
                            0
                            if config.processing.scratch_dir
                            else config.processing.netcdf_compression_level
                        ),
                    )
                    for path_text in saved_segments:
                        path = Path(path_text)
                        segment_paths[path.stem].append(path)

                    if config.figures.enabled:
                        if config.figures.model_by_model:
                            logger.info(
                                "Generating projection figures for %s/%s/%s",
                                model,
                                scenario,
                                period.label,
                            )
                        figure_paths, rows = _projection_diagnostics(
                            config,
                            historical,
                            baseline_corrected,
                            model_inputs,
                            corrected,
                            model=model,
                            scenario=scenario,
                            period=period.label,
                        )
                        manifest["models"][model]["figures"].extend(map(str, figure_paths))
                        model_projection_rows.extend(rows)
                    del corrected, model_inputs
                    gc.collect()

                logger.info("Merging %s/%s into continuous 2015-2100 files", model, scenario)
                manifest["models"][model]["outputs"].extend(
                    _merge_future_segments(
                        config,
                        model=model,
                        scenario=scenario,
                        segment_paths=segment_paths,
                        summary_rows=summary_rows,
                    )
                )
            manifest["models"][model]["status"] = "complete"
            all_evaluation_rows.extend(model_evaluation_rows)
            all_projection_rows.extend(model_projection_rows)
            logger.info("Completed model %s", model)
        except Exception as exc:
            manifest["models"][model]["status"] = "failed"
            manifest["models"][model]["error"] = str(exc)
            logger.exception("Model %s failed", model)
            write_manifest(manifest, manifest_path)
            if not config.processing.continue_on_model_error:
                manifest["status"] = "failed"
                manifest["finished_utc"] = datetime.now(UTC).isoformat()
                write_manifest(manifest, manifest_path)
                raise
        finally:
            gc.collect()
        write_summary(summary_rows, config.run_dir / "summary.csv")
        write_manifest(manifest, manifest_path)

    complete = [item for item in manifest["models"].values() if item["status"] == "complete"]
    if not complete:
        manifest["status"] = "failed"
        write_manifest(manifest, manifest_path)
        raise RuntimeError("No model completed successfully.")

    manifest["status"] = (
        "complete"
        if all(item["status"] == "complete" for item in manifest["models"].values())
        else "partial"
    )
    if config.figures.enabled:
        logger.info("Generating cross-model paper figures")
        figure_paths = make_ensemble_figures(
            all_evaluation_rows,
            all_projection_rows,
            output_dir=config.run_dir / "figures" / "core",
            settings=config.figures,
        )
        manifest["figures"] = [str(path) for path in figure_paths]
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    write_summary(summary_rows, config.run_dir / "summary.csv")
    write_manifest(manifest, manifest_path)
    logger.info("Run finished with status: %s", manifest["status"])
    return manifest
