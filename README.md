# Cloud QDM Climate

[![CI](https://github.com/ZamanRokon/cloud-qdm-climate/actions/workflows/ci.yml/badge.svg)](https://github.com/ZamanRokon/cloud-qdm-climate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)

Cloud-assisted bias adjustment of daily CMIP6 temperature and precipitation
with **Quantile Delta Mapping (QDM)**. The workflow uses a latitude/longitude
bounding box as the area of interest, Google Earth Engine for ERA5-Land,
CHIRPS, and NEX-GDDP-CMIP6, and Google Colab or another Python runtime for the
actual QDM computation.

For precipitation, users can choose either:

- **CHIRPS mode** — daily CHIRPS is queried directly from Earth Engine; or
- **MSWEP mode** — the user supplies an authorized local/Google Drive NetCDF,
  multi-file NetCDF collection, or Zarr store.

No climate data are included in this repository.

## What runs where?

| Location | Responsibility |
|---|---|
| Browser or local PC | Prepare configuration and, for MSWEP mode, provide the licensed file |
| Google Drive | Persistent inputs and outputs for Colab runs |
| Google Earth Engine | Filter and stream ERA5-Land, CHIRPS, and NEX-GDDP-CMIP6 subsets |
| Google Colab/Python | Regrid, train/apply QDM, enforce physical constraints, summarize, and export |

Earth Engine avoids manual downloads of its global archives, but the selected
AOI and time ranges are still transferred to the Python runtime for QDM.

## Scientific workflow

1. Validate `min_lon`, `min_lat`, `max_lon`, and `max_lat`.
2. Load daily reference data for a common calibration period (recommended:
   1981–2014).
3. Retrieve the same dates from each model's `historical` experiment.
4. Regrid historical model data to the corresponding reference grid.
5. Train monthly QDM independently for `tas`, `tasmax`, `tasmin`, and `pr`.
6. Correct the historical period and future windows from 2015–2100.
7. Enforce non-negative precipitation and `tasmin ≤ tas ≤ tasmax`.
8. Save per-variable NetCDF files, trained adjustment parameters, run
   provenance, and summary statistics.

Temperature uses additive QDM; precipitation uses multiplicative QDM with
optional wet-day frequency adaptation. Future data are adjusted in windows
rather than treating 2015–2100 as one stationary distribution.

## Supported online datasets

| Purpose | Earth Engine collection | Variables |
|---|---|---|
| Model historical/future | `NASA/GDDP-CMIP6` | `tas`, `tasmax`, `tasmin`, `pr` |
| Temperature reference | `ECMWF/ERA5_LAND/DAILY_AGGR` | mean/max/min 2 m temperature |
| Optional precipitation reference | `UCSB-CHG/CHIRPS/DAILY` | daily precipitation |

Official catalogs: [NEX-GDDP-CMIP6](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP-CMIP6),
[ERA5-Land Daily Aggregated](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR),
[CHIRPS Daily v2](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY),
and [MSWEP access/documentation](https://www.gloh2o.org/).

The Earth Engine NEX-GDDP-CMIP6 collection supports `ssp245` and `ssp585`.
The input is already statistically downscaled to approximately 0.25°; this
project performs an additional observation-based bias adjustment and grid
alignment, not dynamical downscaling.

## Quick start in Google Colab

Open [`notebooks/cloud_qdm_colab.ipynb`](notebooks/cloud_qdm_colab.ipynb) in
Colab, then:

1. Mount Google Drive.
2. Clone this repository.
3. Install `.[cloud]`.
4. Authenticate Earth Engine.
5. Copy and edit one example configuration.
6. Run `cloud-qdm validate` followed by `cloud-qdm run`.

For MSWEP, place the file in Drive once, for example:

```text
/content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
```

## Local installation

```bash
git clone https://github.com/ZamanRokon/cloud-qdm-climate.git
cd cloud-qdm-climate
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cloud,test]"
```

Validate a configuration without contacting Earth Engine:

```bash
cloud-qdm validate configs/example_chirps.yml
```

Inspect an MSWEP file's variables and coordinates:

```bash
cloud-qdm inspect-mswep /path/to/mswep_daily.nc
```

Run:

```bash
cloud-qdm run configs/example_chirps.yml
```

## Configuration

The AOI is a rectangle in EPSG:4326:

```yaml
aoi:
  min_lon: 89.0
  min_lat: 23.0
  max_lon: 93.0
  max_lat: 26.5
```

Choose the precipitation reference:

```yaml
precipitation_reference:
  mode: chirps
```

or:

```yaml
precipitation_reference:
  mode: mswep
  path: /content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
  variable: precipitation
  latitude_name: lat
  longitude_name: lon
  time_name: time
  unit_scale: 1.0
  aggregate_to_daily: false
```

Complete examples are in [`configs/`](configs/).

## Output layout

```text
outputs/<run-name>/
├── adjustments/<model>/qdm_<variable>.nc
├── corrected/<model>/historical/calibration/<variable>.nc
├── corrected/<model>/<scenario>/<window>/<variable>.nc
├── logs/pipeline.log
├── run-config.yml
├── run-manifest.json
└── summary.csv
```

## Important limitations

- QDM reduces distributional bias; it does not correct every model error or
  create independent fine-scale weather dynamics.
- Results depend strongly on reference quality, calibration period, model
  selection, wet-day treatment, and regridding.
- Independently corrected temperature variables require the included physical
  consistency step.
- Long return periods and impact decisions require separate validation.
- Colab is appropriate for personal and small-team jobs, not a reliable
  unattended multi-user backend.
- MSWEP is not redistributed here and has separate access and licensing terms.

Do not use unvalidated output as the sole basis for safety-critical,
engineering, financial, or regulatory decisions.

## Documentation

- [Technical manual](docs/technical-manual.md)
- [Methodology and validation](docs/methodology.md)
- [Data licensing and governance](docs/data-governance.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

The software and documentation are licensed under Apache-2.0. Dataset licenses
remain separate; see [`NOTICE`](NOTICE) and the data-governance guide.
