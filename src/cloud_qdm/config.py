"""Validated YAML configuration for Cloud QDM Climate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCENARIOS = frozenset({"ssp245", "ssp585"})
SUPPORTED_MODELS = frozenset(
    {
        "ACCESS-CM2",
        "ACCESS-ESM1-5",
        "BCC-CSM2-MR",
        "CESM2",
        "CESM2-WACCM",
        "CMCC-CM2-SR5",
        "CMCC-ESM2",
        "CNRM-CM6-1",
        "CNRM-ESM2-1",
        "CanESM5",
        "EC-Earth3",
        "EC-Earth3-Veg-LR",
        "FGOALS-g3",
        "GFDL-CM4",
        "GFDL-ESM4",
        "GISS-E2-1-G",
        "HadGEM3-GC31-LL",
        "HadGEM3-GC31-MM",
        "IITM-ESM",
        "INM-CM4-8",
        "INM-CM5-0",
        "IPSL-CM6A-LR",
        "KACE-1-0-G",
        "KIOST-ESM",
        "MIROC-ES2L",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "MPI-ESM1-2-LR",
        "MRI-ESM2-0",
        "NESM3",
        "NorESM2-LM",
        "NorESM2-MM",
        "TaiESM1",
        "UKESM1-0-LL",
    }
)


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
    interpolation: str = "linear"
    extrapolation: str = "constant"


@dataclass(frozen=True)
class ProcessingConfig:
    earth_engine_chunk_years: int = 5
    latitude_chunk: int = 40
    longitude_chunk: int = 40
    minimum_time_coverage: float = 0.95
    continue_on_model_error: bool = False
    save_reference_subsets: bool = False


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
    grid_labels: dict[str, str] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir).expanduser() / self.name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
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
        return data


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
            aggregate_to_daily=bool(data.get("aggregate_to_daily", False)),
        )
        if not math.isfinite(mswep.unit_scale) or mswep.unit_scale <= 0:
            raise ConfigurationError("MSWEP unit_scale must be a positive finite number.")
    return PrecipitationReferenceConfig(mode=mode, chirps_collection=collection, mswep=mswep)


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

    run = _section(raw, "run")
    name = str(run.get("name", "")).strip()
    output_dir = str(run.get("output_dir", "")).strip()
    if not name or any(char in name for char in "/\\"):
        raise ConfigurationError("run.name must be a safe non-empty directory name.")
    if not output_dir:
        raise ConfigurationError("run.output_dir is required.")

    ee_raw = _section(raw, "earth_engine")
    project_id = str(ee_raw.get("project_id", "")).strip()
    if not project_id:
        raise ConfigurationError("earth_engine.project_id is required.")
    earth_engine = EarthEngineConfig(
        project_id=project_id,
        era5_collection=str(ee_raw.get("era5_collection", "ECMWF/ERA5_LAND/DAILY_AGGR")),
        gddp_collection=str(ee_raw.get("gddp_collection", "NASA/GDDP-CMIP6")),
    )

    aoi = Bounds.from_mapping(_section(raw, "aoi"))
    calibration = Period.from_mapping(_section(raw, "calibration"), "calibration")
    if calibration.end > date(2014, 12, 31):
        raise ConfigurationError("NEX historical calibration must end on or before 2014-12-31.")

    windows_raw = raw.get("future_windows")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ConfigurationError("future_windows must be a non-empty YAML list.")
    future_windows = tuple(
        Period.from_mapping(item, f"future_windows[{index}]")
        for index, item in enumerate(windows_raw)
        if isinstance(item, dict)
    )
    if len(future_windows) != len(windows_raw):
        raise ConfigurationError("Every future_windows entry must be a mapping.")
    ordered = sorted(future_windows, key=lambda item: item.start)
    for index, item in enumerate(ordered):
        if item.start < date(2015, 1, 1) or item.end > date(2100, 12, 31):
            raise ConfigurationError("Future windows must fall within 2015-01-01 to 2100-12-31.")
        if index and item.start <= ordered[index - 1].end:
            raise ConfigurationError("Future windows must not overlap.")

    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigurationError("models must be a non-empty YAML list.")
    models = tuple(dict.fromkeys(str(item).strip() for item in models_raw))
    unsupported_models = sorted(set(models) - SUPPORTED_MODELS)
    if unsupported_models:
        raise ConfigurationError(
            "Unsupported NEX-GDDP-CMIP6 model ID(s): " + ", ".join(unsupported_models)
        )

    scenarios_raw = raw.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ConfigurationError("scenarios must be a non-empty YAML list.")
    scenarios = tuple(dict.fromkeys(str(item).lower().strip() for item in scenarios_raw))
    unsupported_scenarios = sorted(set(scenarios) - SUPPORTED_SCENARIOS)
    if unsupported_scenarios:
        raise ConfigurationError(
            "Earth Engine NEX-GDDP supports only ssp245 and ssp585; invalid: "
            + ", ".join(unsupported_scenarios)
        )

    grid_labels = {str(key): str(value) for key, value in raw.get("grid_labels", {}).items()}
    if "GFDL-CM4" in models and grid_labels.get("GFDL-CM4") not in {"gr1", "gr2"}:
        raise ConfigurationError("GFDL-CM4 requires grid_labels.GFDL-CM4 set to gr1 or gr2.")

    qdm_raw = raw.get("qdm", {})
    qdm = QDMConfig(
        nquantiles=int(qdm_raw.get("nquantiles", 50)),
        group=str(qdm_raw.get("group", "time.month")),
        wet_day_threshold_mm=float(qdm_raw.get("wet_day_threshold_mm", 0.1)),
        adapt_wet_day_frequency=bool(qdm_raw.get("adapt_wet_day_frequency", True)),
        interpolation=str(qdm_raw.get("interpolation", "linear")),
        extrapolation=str(qdm_raw.get("extrapolation", "constant")),
    )
    if qdm.nquantiles < 5:
        raise ConfigurationError("qdm.nquantiles must be at least 5.")
    if qdm.group != "time.month":
        raise ConfigurationError("Version 0.1 supports qdm.group='time.month' only.")
    if qdm.wet_day_threshold_mm < 0:
        raise ConfigurationError("qdm.wet_day_threshold_mm must be non-negative.")

    processing_raw = raw.get("processing", {})
    processing = ProcessingConfig(
        earth_engine_chunk_years=int(processing_raw.get("earth_engine_chunk_years", 5)),
        latitude_chunk=int(processing_raw.get("latitude_chunk", 40)),
        longitude_chunk=int(processing_raw.get("longitude_chunk", 40)),
        minimum_time_coverage=float(processing_raw.get("minimum_time_coverage", 0.95)),
        continue_on_model_error=bool(processing_raw.get("continue_on_model_error", False)),
        save_reference_subsets=bool(processing_raw.get("save_reference_subsets", False)),
    )
    if processing.earth_engine_chunk_years < 1:
        raise ConfigurationError("processing.earth_engine_chunk_years must be >= 1.")
    if processing.latitude_chunk < 1 or processing.longitude_chunk < 1:
        raise ConfigurationError("Spatial chunks must be positive integers.")
    if not 0 < processing.minimum_time_coverage <= 1:
        raise ConfigurationError("minimum_time_coverage must be in (0, 1].")

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
        qdm=qdm,
        processing=processing,
        grid_labels=grid_labels,
    )


def dump_config(config: RunConfig, path: str | Path) -> None:
    """Write a normalized configuration with stable key ordering."""
    Path(path).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
