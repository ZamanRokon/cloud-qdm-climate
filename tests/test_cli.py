from pathlib import Path

import tomllib

from cloud_qdm.cli import build_parser


def test_primary_command_is_qdm() -> None:
    assert build_parser().prog == "qdm"
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"qdm": "cloud_qdm.cli:main"}
