from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

from cloud_qdm.cli import build_parser


def test_primary_command_is_qdm() -> None:
    assert build_parser().prog == "qdm"
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"qdm": "cloud_qdm.cli:main"}
