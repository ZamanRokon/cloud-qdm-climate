"""Cloud-assisted Quantile Delta Mapping for daily climate projections."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cloud-qdm-climate")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
