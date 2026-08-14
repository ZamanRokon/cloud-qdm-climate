"""End-to-end model-by-model QDM pipeline."""

from __future__ import annotations

import gc
from datetime import UTC, datetime
from typing import Any

import yaml

from cloud_qdm.config import RunConfig
from cloud_qdm.coordinates import (
    align_calibration_time,
    enforce_temperature_ordering,
    regrid_to_reference,
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
    summarize,
    write_manifest,
    write_summary,
)
from cloud_qdm.sources import (
    VARIABLES,
    fetch_chirps_reference,
    fetch_era5_reference,
    fetch_gddp,
    initialize_earth_engine,
    open_mswep,
)

DataByVariable = dict[str, Any]
EvaluationData = tuple[DataByVariable, DataByVariable, DataByVariable]


def _load_references(config: RunConfig, logger) -> DataByVariable:
    references = {}
    for variable in ("tas", "tasmax", "tasmin"):
        logger.info("Loading ERA5-Land reference: %s", variable)
        references[variable] = fetch_era5_reference(
            collection_id=config.earth_engine.era5_collection,
            variable=variable,
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
        bounds=config.aoi,
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
    for variable in VARIABLES:
        if preloaded is not None:
            model_on_reference = preloaded[variable]
        else:
            raw = _fetch_model(
                config, model=model, scenario=scenario, period=period, variable=variable
            )
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
) -> list[str]:
    output_paths = []
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
    for variable, data in corrected.items():
        output_path = (
            config.run_dir / "corrected" / model / scenario / period.label / f"{variable}.nc"
        )
        save_netcdf(
            data,
            output_path,
            {**provenance, "reference_source": _reference_source(config, variable)},
        )
        summary_rows.append(
            summarize(
                data,
                model=model,
                scenario=scenario,
                period=period.label,
                variable=variable,
                output_path=output_path,
            )
        )
        output_paths.append(str(output_path))
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
    for variable in VARIABLES:
        logger.info("Fetching historical %s/%s", model, variable)
        historical = _fetch_model(
            config,
            model=model,
            scenario="historical",
            period=config.calibration,
            variable=variable,
        )
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
            )

    summary_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
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
                logger.info("Generating independent evaluation figures for %s", model)
                figure_paths, model_evaluation_rows = make_evaluation_figures(
                    *evaluation,
                    model=model,
                    period=config.evaluation.validation.label,
                    output_dir=config.run_dir / "figures" / "by-model" / model / "evaluation",
                    settings=config.figures,
                    wet_day_threshold=config.qdm.wet_day_threshold_mm,
                )
                manifest["models"][model]["figures"].extend(map(str, figure_paths))

            periods = [("historical", config.calibration, historical)]
            periods.extend(
                (scenario, period, None)
                for scenario in config.scenarios
                for period in config.future_windows
            )
            baseline_corrected = None
            for scenario, period, preloaded in periods:
                logger.info("Correcting %s/%s/%s", model, scenario, period.label)
                corrected, model_inputs = _correct_period(
                    config,
                    model=model,
                    scenario=scenario,
                    period=period,
                    references=references,
                    adjustments=adjustments,
                    preloaded=preloaded,
                )
                manifest["models"][model]["outputs"].extend(
                    _save_period(
                        config,
                        model=model,
                        scenario=scenario,
                        period=period,
                        corrected=corrected,
                        summary_rows=summary_rows,
                    )
                )
                if scenario == "historical":
                    baseline_corrected = corrected
                elif config.figures.enabled:
                    if baseline_corrected is None:
                        raise RuntimeError("Historical baseline was not available for figures.")
                    figure_paths, rows = make_projection_figures(
                        historical,
                        baseline_corrected,
                        model_inputs,
                        corrected,
                        model=model,
                        scenario=scenario,
                        period=period.label,
                        output_dir=(
                            config.run_dir
                            / "figures"
                            / "by-model"
                            / model
                            / "projection"
                            / scenario
                            / period.label
                        ),
                        settings=config.figures,
                    )
                    manifest["models"][model]["figures"].extend(map(str, figure_paths))
                    model_projection_rows.extend(rows)
                del corrected
                gc.collect()
            manifest["models"][model]["status"] = "complete"
            evaluation_rows.extend(model_evaluation_rows)
            projection_rows.extend(model_projection_rows)
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
            evaluation_rows,
            projection_rows,
            output_dir=config.run_dir / "figures" / "core",
            settings=config.figures,
        )
        manifest["figures"] = [str(path) for path in figure_paths]
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    write_summary(summary_rows, config.run_dir / "summary.csv")
    write_manifest(manifest, manifest_path)
    logger.info("Run finished with status: %s", manifest["status"])
    return manifest
