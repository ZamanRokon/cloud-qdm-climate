"""Validated YAML configuration for Cloud QDM Climate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCENARIOS = frozenset({"ssp245", "ssp585"})
_MODEL_IDS = """
ACCESS-CM2 ACCESS-ESM1-5 BCC-CSM2-MR CESM2 CESM2-WACCM
CMCC-CM2-SR5 CMCC-ESM2 CNRM-CM6-1 CNRM-ESM2-1 CanESM5
EC-Earth3 EC-Earth3-Veg-LR FGOALS-g3 GFDL-CM4 GFDL-ESM4
GISS-E2-1-G HadGEM3-GC31-LL HadGEM3-GC31-MM IITM-ESM INM-CM4-8
INM-CM5-0 IPSL-CM6A-LR KACE-1-0-G KIOST-ESM MIROC-ES2L MIROC6
MPI-ESM1-2-HR MPI-ESM1-2-LR MRI-ESM2-0 NESM3 NorESM2-LM
NorESM2-MM TaiESM1 UKESM1-0-LL
"""
SUPPORTED_MODELS = frozenset(_MODEL_IDS.split())


class ConfigurationError(ValueError):
    """Raised when a run configuration is invalid."""


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{name}' must be a YAML mapping.")
    return value


def _iso(value: Any, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigurationError(f"'{field_name}' must use YYYY-MM-DD format.") from exc


@dataclass(frozen=True)
class Bounds:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Bounds:
        try:
            bounds = cls(
                min_lon=float(data["min_lon"]),
                min_lat=float(data["min_lat"]),
                max_lon=float(data["max_lon"]),
                max_lat=float(data["max_lat"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                "AOI requires numeric min_lon, min_lat, max_lon, and max_lat."
            ) from exc
        bounds.validate()
        return bounds

    def validate(self) -> None:
        values = (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("AOI coordinates must be finite numbers.")
        if not (-180 <= self.min_lon < self.max_lon <= 180):
            raise ConfigurationError(
                "Longitude must satisfy -180 <= min_lon < max_lon <= 180; "
                "antimeridian crossing is not supported."
            )
        if not (-90 <= self.min_lat < self.max_lat <= 90):
            raise ConfigurationError("Latitude must satisfy -90 <= min_lat < max_lat <= 90.")

    def as_list(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    def padded(self, degrees: float) -> Bounds:
        """Expand bounds without crossing valid longitude/latitude limits."""
        return Bounds(
            min_lon=max(-180.0, self.min_lon - degrees),
            min_lat=max(-90.0, self.min_lat - degrees),
            max_lon=min(180.0, self.max_lon + degrees),
            max_lat=min(90.0, self.max_lat + degrees),
        )


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any], field_name: str) -> Period:
        start = _iso(data.get("start"), f"{field_name}.start")
        end = _iso(data.get("end"), f"{field_name}.end")
        label = str(data.get("label", f"{start.year}-{end.year}")).strip()
        if start > end:
            raise ConfigurationError(f"'{field_name}' start must not be after end.")
        if not label or any(char in label for char in "/\\"):
            raise ConfigurationError(f"'{field_name}.label' must be a safe non-empty name.")
        return cls(start=start, end=end, label=label)


@dataclass(frozen=True)
class EarthEngineConfig:
    project_id: str
    era5_collection: str = "ECMWF/ERA5_LAND/DAILY_AGGR"
    gddp_collection: str = "NASA/GDDP-CMIP6"


@dataclass(frozen=True)
class MSWEPConfig:
    path: str
    variable: str = "precipitation"
    latitude_name: str = "lat"
    longitude_name: str = "lon"
    time_name: str = "time"
    unit_scale: float = 1.0
    aggregate_to_daily: bool = False
    fill_non_finite_with_zero: bool = False


@dataclass(frozen=True)
class PrecipitationReferenceConfig:
    mode: str
    chirps_collection: str = "UCSB-CHG/CHIRPS/DAILY"
    mswep: MSWEPConfig | None = None


@dataclass(frozen=True)
class QDMConfig:
    nquantiles: int = 50
    group: str = "time.month"
    wet_day_threshold_mm: float = 0.1
    adapt_wet_day_frequency: bool = True
    random_seed: int = 42
    interpolation: str = "linear"
    extrapolation: str = "constant"


@dataclass(frozen=True)
class ProcessingConfig:
    earth_engine_chunk_years: int = 10
    latitude_chunk: int = 40
    longitude_chunk: int = 40
    netcdf_compression_level: int = 1
    scratch_dir: str | None = None
    minimum_time_coverage: float = 0.95
    continue_on_model_error: bool = False
    save_reference_subsets: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    training: Period
    validation: Period


@dataclass(frozen=True)
class FigureConfig:
    enabled: bool = False
    model_by_model: bool = False
    formats: tuple[str, ...] = ("png",)
    dpi: int = 600


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    earth_engine: EarthEngineConfig
    aoi: Bounds
    calibration: Period
    future_windows: tuple[Period, ...]
    models: tuple[str, ...]
    scenarios: tuple[str, ...]
    precipitation_reference: PrecipitationReferenceConfig
    qdm: QDMConfig = field(default_factory=QDMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    evaluation: EvaluationConfig | None = None
    figures: FigureConfig = field(default_factory=FigureConfig)
    grid_labels: dict[str, str] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir).expanduser() / self.name

    @property
    def segment_dir(self) -> Path:
        """Return the disposable future-window workspace."""
        if self.processing.scratch_dir:
            return Path(self.processing.scratch_dir).expanduser() / self.name / ".segments"
        return self.run_dir / ".segments"

    def to_dict(self) -> dict[str, Any]:
        data = {"run": {"name": self.name, "output_dir": self.output_dir}, **asdict(self)}
        data.pop("name")
        data.pop("output_dir")
        data["calibration"] = {
            "start": self.calibration.start.isoformat(),
            "end": self.calibration.end.isoformat(),
            "label": self.calibration.label,
        }
        data["future_windows"] = [
            {"start": item.start.isoformat(), "end": item.end.isoformat(), "label": item.label}
            for item in self.future_windows
        ]
        data["models"] = list(self.models)
        data["scenarios"] = list(self.scenarios)
        data["precipitation_reference"] = {
            "mode": self.precipitation_reference.mode,
            "chirps_collection": self.precipitation_reference.chirps_collection,
        }
        if self.precipitation_reference.mswep:
            data["precipitation_reference"].update(asdict(self.precipitation_reference.mswep))
        data["figures"]["formats"] = list(self.figures.formats)
        if self.evaluation:
            data["evaluation"] = {
                "training": _period_mapping(self.evaluation.training),
                "validation": _period_mapping(self.evaluation.validation),
            }
        return data


def _required_text(data: dict[str, Any], key: str, field_name: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ConfigurationError(f"{field_name} is required.")
    return value


def _period_mapping(period: Period) -> dict[str, str]:
    return {
        "start": period.start.isoformat(),
        "end": period.end.isoformat(),
        "label": period.label,
    }


def _boolean(data: dict[str, Any], key: str, default: bool, section: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{section}.{key} must be true or false.")
    return value


def _unique_list(raw: dict[str, Any], key: str, *, lowercase: bool = False) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise ConfigurationError(f"{key} must be a non-empty YAML list.")
    cleaned = [str(value).strip() for value in values]
    if lowercase:
        cleaned = [value.lower() for value in cleaned]
    if any(not value for value in cleaned):
        raise ConfigurationError(f"{key} entries must be non-empty strings.")
    return tuple(dict.fromkeys(cleaned))


def _parse_run(raw: dict[str, Any]) -> tuple[str, str]:
    data = _section(raw, "run")
    name = _required_text(data, "name", "run.name")
    if any(char in name for char in "/\\"):
        raise ConfigurationError("run.name must be a safe directory name.")
    return name, _required_text(data, "output_dir", "run.output_dir")


def _parse_earth_engine(raw: dict[str, Any]) -> EarthEngineConfig:
    data = _section(raw, "earth_engine")
    return EarthEngineConfig(
        project_id=_required_text(data, "project_id", "earth_engine.project_id"),
        era5_collection=str(data.get("era5_collection", "ECMWF/ERA5_LAND/DAILY_AGGR")),
        gddp_collection=str(data.get("gddp_collection", "NASA/GDDP-CMIP6")),
    )


def _parse_periods(raw: dict[str, Any]) -> tuple[Period, tuple[Period, ...]]:
    calibration = Period.from_mapping(_section(raw, "calibration"), "calibration")
    if calibration.end > date(2014, 12, 31):
        raise ConfigurationError("NEX historical calibration must end on or before 2014-12-31.")

    values = raw.get("future_windows")
    if not isinstance(values, list) or not values:
        raise ConfigurationError("future_windows must be a non-empty YAML list.")
    if not all(isinstance(value, dict) for value in values):
        raise ConfigurationError("Every future_windows entry must be a mapping.")

    windows = tuple(
        Period.from_mapping(value, f"future_windows[{index}]") for index, value in enumerate(values)
    )
    ordered = sorted(windows, key=lambda item: item.start)
    for previous, current in pairwise(ordered):
        if current.start <= previous.end:
            raise ConfigurationError("Future windows must not overlap.")
        if current.start != previous.end + timedelta(days=1):
            raise ConfigurationError("Future windows must be continuous without date gaps.")
    if any(
        window.start < date(2015, 1, 1) or window.end > date(2100, 12, 31) for window in windows
    ):
        raise ConfigurationError("Future windows must fall within 2015-01-01 to 2100-12-31.")
    if ordered[0].start != date(2015, 1, 1) or ordered[-1].end != date(2100, 12, 31):
        raise ConfigurationError("Future windows must collectively cover 2015-01-01 to 2100-12-31.")
    return calibration, tuple(ordered)


def _parse_models_and_scenarios(
    raw: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    models = _unique_list(raw, "models")
    unsupported_models = sorted(set(models) - SUPPORTED_MODELS)
    if unsupported_models:
        raise ConfigurationError(
            "Unsupported NEX-GDDP-CMIP6 model ID(s): " + ", ".join(unsupported_models)
        )

    scenarios = _unique_list(raw, "scenarios", lowercase=True)
    unsupported_scenarios = sorted(set(scenarios) - SUPPORTED_SCENARIOS)
    if unsupported_scenarios:
        raise ConfigurationError(
            "Earth Engine NEX-GDDP supports only ssp245 and ssp585; invalid: "
            + ", ".join(unsupported_scenarios)
        )
    return models, scenarios


def _parse_precipitation(data: dict[str, Any]) -> PrecipitationReferenceConfig:
    mode = str(data.get("mode", "")).lower().strip()
    if mode not in {"chirps", "mswep"}:
        raise ConfigurationError("precipitation_reference.mode must be 'chirps' or 'mswep'.")
    collection = str(data.get("chirps_collection", "UCSB-CHG/CHIRPS/DAILY")).strip()
    mswep = None
    if mode == "mswep":
        path = str(data.get("path", "")).strip()
        if not path:
            raise ConfigurationError("MSWEP mode requires precipitation_reference.path.")
        mswep = MSWEPConfig(
            path=path,
            variable=str(data.get("variable", "precipitation")),
            latitude_name=str(data.get("latitude_name", "lat")),
            longitude_name=str(data.get("longitude_name", "lon")),
            time_name=str(data.get("time_name", "time")),
            unit_scale=float(data.get("unit_scale", 1.0)),
            aggregate_to_daily=_boolean(
                data, "aggregate_to_daily", False, "precipitation_reference"
            ),
            fill_non_finite_with_zero=_boolean(
                data,
                "fill_non_finite_with_zero",
                False,
                "precipitation_reference",
            ),
        )
        if not math.isfinite(mswep.unit_scale) or mswep.unit_scale <= 0:
            raise ConfigurationError("MSWEP unit_scale must be a positive finite number.")
    return PrecipitationReferenceConfig(mode=mode, chirps_collection=collection, mswep=mswep)


def _parse_qdm(raw: dict[str, Any]) -> QDMConfig:
    data = raw.get("qdm", {})
    if not isinstance(data, dict):
        raise ConfigurationError("'qdm' must be a YAML mapping.")
    try:
        config = QDMConfig(
            nquantiles=int(data.get("nquantiles", 50)),
            group=str(data.get("group", "time.month")),
            wet_day_threshold_mm=float(data.get("wet_day_threshold_mm", 0.1)),
            adapt_wet_day_frequency=_boolean(data, "adapt_wet_day_frequency", True, "qdm"),
            random_seed=int(data.get("random_seed", 42)),
            interpolation=str(data.get("interpolation", "linear")),
            extrapolation=str(data.get("extrapolation", "constant")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("qdm contains an invalid numeric value.") from exc
    if config.nquantiles < 5:
        raise ConfigurationError("qdm.nquantiles must be at least 5.")
    if config.group != "time.month":
        raise ConfigurationError("Version 0.1 supports qdm.group='time.month' only.")
    if config.wet_day_threshold_mm < 0:
        raise ConfigurationError("qdm.wet_day_threshold_mm must be non-negative.")
    if not 0 <= config.random_seed <= 2**32 - 1:
        raise ConfigurationError("qdm.random_seed must be between 0 and 4294967295.")
    return config


def _parse_processing(raw: dict[str, Any]) -> ProcessingConfig:
    data = raw.get("processing", {})
    if not isinstance(data, dict):
        raise ConfigurationError("'processing' must be a YAML mapping.")
    scratch_dir = data.get("scratch_dir")
    if scratch_dir is not None and (not isinstance(scratch_dir, str) or not scratch_dir.strip()):
        raise ConfigurationError("processing.scratch_dir must be a non-empty path or null.")
    try:
        config = ProcessingConfig(
            earth_engine_chunk_years=int(data.get("earth_engine_chunk_years", 10)),
            latitude_chunk=int(data.get("latitude_chunk", 40)),
            longitude_chunk=int(data.get("longitude_chunk", 40)),
            netcdf_compression_level=int(data.get("netcdf_compression_level", 1)),
            scratch_dir=(scratch_dir.strip() if scratch_dir else None),
            minimum_time_coverage=float(data.get("minimum_time_coverage", 0.95)),
            continue_on_model_error=_boolean(data, "continue_on_model_error", False, "processing"),
            save_reference_subsets=_boolean(data, "save_reference_subsets", False, "processing"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("processing contains an invalid numeric value.") from exc
    if config.earth_engine_chunk_years < 1:
        raise ConfigurationError("processing.earth_engine_chunk_years must be >= 1.")
    if min(config.latitude_chunk, config.longitude_chunk) < 1:
        raise ConfigurationError("Spatial chunks must be positive integers.")
    if not 0 <= config.netcdf_compression_level <= 9:
        raise ConfigurationError("processing.netcdf_compression_level must be between 0 and 9.")
    if not 0 < config.minimum_time_coverage <= 1:
        raise ConfigurationError("minimum_time_coverage must be in (0, 1].")
    return config


def _parse_evaluation(raw: dict[str, Any], calibration: Period) -> EvaluationConfig | None:
    data = raw.get("evaluation")
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConfigurationError("'evaluation' must be a YAML mapping.")
    training = Period.from_mapping(_section(data, "training"), "evaluation.training")
    validation = Period.from_mapping(_section(data, "validation"), "evaluation.validation")
    for name, period in (("training", training), ("validation", validation)):
        if period.start < calibration.start or period.end > calibration.end:
            raise ConfigurationError(f"evaluation.{name} must fall inside calibration.")
    if training.end >= validation.start:
        raise ConfigurationError(
            "Evaluation training must end before the validation period starts."
        )
    return EvaluationConfig(training=training, validation=validation)


def _parse_figures(raw: dict[str, Any]) -> FigureConfig:
    data = raw.get("figures", {})
    if not isinstance(data, dict):
        raise ConfigurationError("'figures' must be a YAML mapping.")
    formats = _unique_list({"formats": data.get("formats", ["png"])}, "formats")
    unsupported = sorted(set(formats) - {"pdf", "png", "svg"})
    if unsupported:
        raise ConfigurationError("figures.formats supports only png, pdf, and svg.")
    try:
        dpi = int(data.get("dpi", 600))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("figures.dpi must be an integer.") from exc
    if not 150 <= dpi <= 600:
        raise ConfigurationError("figures.dpi must be between 150 and 600.")
    return FigureConfig(
        enabled=_boolean(data, "enabled", False, "figures"),
        model_by_model=_boolean(data, "model_by_model", False, "figures"),
        formats=formats,
        dpi=dpi,
    )


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run configuration."""
    config_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a YAML mapping.")

    name, output_dir = _parse_run(raw)
    earth_engine = _parse_earth_engine(raw)
    aoi = Bounds.from_mapping(_section(raw, "aoi"))
    calibration, future_windows = _parse_periods(raw)
    models, scenarios = _parse_models_and_scenarios(raw)

    labels = raw.get("grid_labels", {})
    if not isinstance(labels, dict):
        raise ConfigurationError("'grid_labels' must be a YAML mapping.")
    grid_labels = {str(key): str(value) for key, value in labels.items()}
    if "GFDL-CM4" in models and grid_labels.get("GFDL-CM4") not in {"gr1", "gr2"}:
        raise ConfigurationError("GFDL-CM4 requires grid_labels.GFDL-CM4 set to gr1 or gr2.")

    evaluation = _parse_evaluation(raw, calibration)
    figures = _parse_figures(raw)
    if figures.enabled and evaluation is None:
        raise ConfigurationError(
            "figures.enabled requires independent evaluation.training and evaluation.validation periods."
        )

    return RunConfig(
        name=name,
        output_dir=output_dir,
        earth_engine=earth_engine,
        aoi=aoi,
        calibration=calibration,
        future_windows=future_windows,
        models=models,
        scenarios=scenarios,
        precipitation_reference=_parse_precipitation(_section(raw, "precipitation_reference")),
        qdm=_parse_qdm(raw),
        processing=_parse_processing(raw),
        evaluation=evaluation,
        figures=figures,
        grid_labels=grid_labels,
    )


def dump_config(config: RunConfig, path: str | Path) -> None:
    """Write a normalized configuration with stable key ordering."""
    Path(path).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
