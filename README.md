# Cloud QDM Climate

[![CI](https://github.com/ZamanRokon/cloud-qdm-climate/actions/workflows/ci.yml/badge.svg)](https://github.com/ZamanRokon/cloud-qdm-climate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)

Cloud-assisted bias adjustment of daily NEX-GDDP-CMIP6 `tas`, `tasmax`,
`tasmin`, and `pr` with monthly Quantile Delta Mapping (QDM). Define a
rectangular latitude/longitude area of interest (AOI); the workflow retrieves
only the requested Earth Engine subsets and performs QDM in Google Colab or
another Python runtime.

This is statistical bias adjustment and grid alignment, not dynamical
downscaling. No climate data are distributed with the repository.

## Choose a precipitation reference

| Mode | Input | Best fit |
|---|---|---|
| CHIRPS | Streamed from `UCSB-CHG/CHIRPS/DAILY` | Fully cloud-based run |
| MSWEP | Your authorized NetCDF files or Zarr store | Work that requires MSWEP |

Both modes use ERA5-Land for daily mean, maximum, and minimum temperature.
NEX-GDDP-CMIP6 supplies historical and future model data. Earth Engine avoids
manual global-archive downloads, but AOI subsets are still transferred to the
Python runtime.

## Quick start

For Colab, open
[`notebooks/cloud_qdm_colab.ipynb`](notebooks/cloud_qdm_colab.ipynb) and run its
cells in order. For Linux or WSL:

```bash
git clone https://github.com/ZamanRokon/cloud-qdm-climate.git
cd cloud-qdm-climate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cloud]"
```

Then:

1. Copy [`configs/example_chirps.yml`](configs/example_chirps.yml) or
   [`configs/example_mswep.yml`](configs/example_mswep.yml).
2. Set the Earth Engine project, output directory, AOI, models, scenarios, and
   periods.
3. In MSWEP mode, set the file path, variable name, coordinate names, units
   scale, and daily-aggregation option.
4. Authenticate Earth Engine once in the runtime.
5. Validate, then run:

```bash
cloud-qdm validate my-run.yml
cloud-qdm run my-run.yml
```

Inspect an unfamiliar MSWEP file before configuring it:

```bash
cloud-qdm inspect-mswep /path/to/mswep_daily.nc
```

The [technical manual](docs/technical-manual.md) gives the exact Colab setup,
configuration field reference, operating guidance, and troubleshooting.

## Implemented method

| Variable | Reference | QDM form | Working units | Post-processing |
|---|---|---|---|---|
| `tas` | ERA5-Land | additive | K | temperature ordering |
| `tasmax` | ERA5-Land | additive | K | temperature ordering |
| `tasmin` | ERA5-Land | additive | K | temperature ordering |
| `pr` | CHIRPS or MSWEP | multiplicative | mm d-1 | wet-day adaptation; clip at zero |

QDM is trained independently by model, variable, grid cell, and calendar
month. Historical model fields are bilinearly interpolated to each reference
grid before training. The fitted parameters are reused for configured future
windows. See [methodology and validation](docs/methodology.md) for equations,
assumptions, and required evaluation.

## Output

```text
outputs/<run-name>/
|-- adjustments/<model>/qdm_<variable>.nc
|-- corrected/<model>/historical/calibration/<variable>.nc
|-- corrected/<model>/<scenario>/<window>/<variable>.nc
|-- logs/pipeline.log
|-- run-config.yml
|-- run-manifest.json
`-- summary.csv
```

The normalized configuration, manifest, saved adjustment parameters, NetCDF
attributes, and summary table provide the run audit trail.

## Know before use

- QDM corrects marginal distributions; it does not repair every model error or
  preserve multivariable and spatial dependence by itself.
- A finer reference grid does not create independent fine-scale dynamics.
- Bilinear precipitation regridding is not conservative.
- Extremes, wet-day behavior, trend preservation, and held-out historical
  performance require study-specific validation.
- Colab is suitable for interactive AOI-scale work, not an unattended service.

Do not use unvalidated output as the sole basis for safety-critical,
engineering, financial, or regulatory decisions.

## Documentation and license

- [Technical manual](docs/technical-manual.md) — setup, configuration, running,
  outputs, and troubleshooting
- [Methodology and validation](docs/methodology.md) — algorithm, assumptions,
  limitations, and evaluation
- [Data licensing and governance](docs/data-governance.md) — provider terms,
  attribution, and credentials

Code and documentation are Apache-2.0 licensed. Dataset terms remain separate;
see [`NOTICE`](NOTICE).
