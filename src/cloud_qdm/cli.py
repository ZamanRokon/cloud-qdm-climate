"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from cloud_qdm.config import ConfigurationError, load_config
from cloud_qdm.pipeline import run_pipeline
from cloud_qdm.sources import inspect_mswep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloud-qdm",
        description="Bias-adjust NEX-GDDP-CMIP6 climate projections with QDM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a YAML configuration.")
    validate.add_argument("config")

    run = subparsers.add_parser("run", help="Execute a validated pipeline configuration.")
    run.add_argument("config")

    inspect = subparsers.add_parser(
        "inspect-mswep", help="Print variables, coordinates, dimensions, and metadata."
    )
    inspect.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-mswep":
            print(inspect_mswep(args.path))
            return 0
        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps(config.to_dict(), indent=2, default=str))
            print("Configuration is valid.")
            return 0
        manifest = run_pipeline(config)
        print(json.dumps(manifest, indent=2))
        return 0 if manifest["status"] == "complete" else 2
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
