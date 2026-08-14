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


def _load_references(config: RunConfig, logger) -> dict[str, Any]:
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
    references: dict[str, Any],
    adjustments: dict[str, Any],
    preloaded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corrected = {}
    for variable in VARIABLES:
        if preloaded:
            model_on_reference = preloaded[variable]
        else:
            raw = _fetch_model(
                config, model=model, scenario=scenario, period=period, variable=variable
            )
            model_on_reference = regrid_to_reference(raw, references[variable])
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
    return corrected


def _save_period(
    config: RunConfig,
    *,
    model: str,
    scenario: str,
    period,
    corrected: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> list[str]:
    output_paths = []
    for variable, data in corrected.items():
        if variable == "pr":
            reference_source = (
                config.precipitation_reference.chirps_collection
                if config.precipitation_reference.mode == "chirps"
                else "MSWEP (user supplied; local path withheld from NetCDF metadata)"
            )
        else:
            reference_source = config.earth_engine.era5_collection
        output_path = (
            config.run_dir / "corrected" / model / scenario / period.label / f"{variable}.nc"
        )
        save_netcdf(
            data,
            output_path,
            {
                "model": model,
                "scenario": scenario,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "calibration_start": config.calibration.start.isoformat(),
                "calibration_end": config.calibration.end.isoformat(),
                "aoi_bounds": config.aoi.as_list(),
                "precipitation_reference": config.precipitation_reference.mode,
                "model_source_collection": config.earth_engine.gddp_collection,
                "reference_source": reference_source,
            },
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


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    """Execute a validated run and return its manifest."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(config.run_dir)
    manifest_path = config.run_dir / "run-manifest.json"
    manifest = base_manifest()
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
    for model in config.models:
        logger.info("Processing model %s", model)
        manifest["models"][model] = {"status": "running", "outputs": [], "error": None}
        write_manifest(manifest, manifest_path)
        try:
            adjustments = {}
            historical_on_reference = {}
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
                logger.info("Training QDM %s/%s", model, variable)
                adjustments[variable] = train_qdm(
                    reference,
                    historical,
                    variable=variable,
                    config=config.qdm,
                )
                adjustment_path = config.run_dir / "adjustments" / model / f"qdm_{variable}.nc"
                adjustment_path.parent.mkdir(parents=True, exist_ok=True)
                save_adjustment(adjustments[variable], str(adjustment_path))

            corrected_historical = _correct_period(
                config,
                model=model,
                scenario="historical",
                period=config.calibration,
                references=references,
                adjustments=adjustments,
                preloaded=historical_on_reference,
            )
            manifest["models"][model]["outputs"].extend(
                _save_period(
                    config,
                    model=model,
                    scenario="historical",
                    period=config.calibration,
                    corrected=corrected_historical,
                    summary_rows=summary_rows,
                )
            )
            del corrected_historical

            for scenario in config.scenarios:
                for period in config.future_windows:
                    logger.info("Correcting %s/%s/%s", model, scenario, period.label)
                    corrected = _correct_period(
                        config,
                        model=model,
                        scenario=scenario,
                        period=period,
                        references=references,
                        adjustments=adjustments,
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
                    del corrected
                    gc.collect()

            manifest["models"][model]["status"] = "complete"
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
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    write_summary(summary_rows, config.run_dir / "summary.csv")
    write_manifest(manifest, manifest_path)
    logger.info("Run finished with status: %s", manifest["status"])
    return manifest
