# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Publication figure suite with held-out evaluation, Taylor diagrams, skill and
  distribution diagnostics, spatial maps, extremes, future change analysis,
  multi-model summaries, and CSV metric exports.
- A paper configuration template and figure interpretation/caption guide.

### Changed

- Simplified configuration parsing and removed repeated correction/provenance logic.
- Reorganized documentation into concise orientation, operations, methodology,
  and data-governance guides.
- Expanded the scientific manual with explicit QDM equations, assumptions, and
  validation requirements.
- Made precipitation wet-day randomization reproducible with `qdm.random_seed`.

## [0.1.0] - 2026-08-14

### Added

- Bounds-based AOI configuration.
- CHIRPS and user-supplied MSWEP precipitation reference modes.
- ERA5-Land temperature references and NEX-GDDP-CMIP6 model retrieval.
- Monthly additive/multiplicative Quantile Delta Mapping.
- Historical and windowed future correction with NetCDF outputs.
- Colab runner, validation CLI, tests, CI, technical manual, and data-governance guidance.
