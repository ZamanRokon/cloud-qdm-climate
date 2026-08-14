from pathlib import Path

import pytest
import yaml

from cloud_qdm.config import Bounds, ConfigurationError, load_config


def test_bounds_validation() -> None:
    Bounds(89, 23, 93, 26.5).validate()
    with pytest.raises(ConfigurationError, match="Longitude"):
        Bounds(93, 23, 89, 26.5).validate()
    with pytest.raises(ConfigurationError, match="Latitude"):
        Bounds(89, -91, 93, 26.5).validate()


def test_example_chirps_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    config = load_config(path)
    assert config.precipitation_reference.mode == "chirps"
    assert config.aoi.as_list() == [89.0, 23.0, 93.0, 26.5]
    assert config.scenarios == ("ssp245", "ssp585")
    assert config.future_windows[-1].end.year == 2100


def test_unsupported_scenario_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["scenarios"] = ["ssp370"]
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="ssp245 and ssp585"):
        load_config(path)


def test_gfdl_cm4_requires_grid_label(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["models"] = ["GFDL-CM4"]
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="grid_labels"):
        load_config(path)


def test_paper_config_has_independent_evaluation() -> None:
    path = Path(__file__).parents[1] / "configs" / "example_paper.yml"
    config = load_config(path)
    assert config.figures.enabled
    assert config.figures.formats == ("png", "pdf")
    assert config.evaluation is not None
    assert config.evaluation.training.end.year == 2004
    assert config.evaluation.validation.start.year == 2005
    assert config.qdm.random_seed == 42


def test_evaluation_must_be_chronological(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_paper.yml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["evaluation"]["training"], raw["evaluation"]["validation"] = (
        raw["evaluation"]["validation"],
        raw["evaluation"]["training"],
    )
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="training must end before"):
        load_config(path)


def test_qdm_random_seed_range_is_validated(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["qdm"]["random_seed"] = -1
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="random_seed"):
        load_config(path)


def test_figures_require_evaluation_periods(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "example_chirps.yml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["figures"] = {"enabled": True}
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="requires independent evaluation"):
        load_config(path)
