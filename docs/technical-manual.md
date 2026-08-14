# Technical manual

## 1. Purpose and scope

This manual describes installation, configuration, execution, output checking,
and troubleshooting. The workflow bias-adjusts daily `tas`, `tasmax`, `tasmin`,
and `pr` from NEX-GDDP-CMIP6 for a rectangular latitude/longitude AOI.

The recommended calibration period is 1981-01-01 through 2014-12-31. Future
projections begin on 2015-01-01 and are processed in multi-decadal windows.

## 2. Runtime architecture

Earth Engine hosts ERA5-Land, CHIRPS, and NEX-GDDP-CMIP6. The Python runtime
constructs filtered Earth Engine requests and xee exposes the returned subsets
as xarray objects. QDM is then trained and applied by xsdba in Colab or another
Python machine. Outputs are written to the configured directory.

For Colab, Google Drive is storage, not the compute engine. Keep the notebook
open during a run. A disconnected runtime loses `/content` but not files already
written to Drive.

## 3. Prerequisites

- Python 3.10–3.13.
- A Google account and Earth Engine-enabled Cloud project.
- Enough Colab/VM memory for the AOI and selected model count.
- For MSWEP mode, authorized access and a supported NetCDF/Zarr copy.

Never place Google access tokens, GitHub tokens, or service-account JSON in the
repository or YAML configuration.

## 4. Installation

### Google Colab

Use the supplied notebook or execute:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/ZamanRokon/cloud-qdm-climate.git
%cd cloud-qdm-climate
!python -m pip install -e ".[cloud]"
```

Authenticate Earth Engine:

```python
import ee

ee.Authenticate()
ee.Initialize(project="your-earth-engine-project")
```

### Linux/WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cloud,test]"
```

Cartopy, GDAL, and geopandas are intentionally unnecessary because the AOI is
a simple EPSG:4326 rectangle.

## 5. AOI configuration

Coordinates must be decimal degrees:

```yaml
aoi:
  min_lon: 89.0
  min_lat: 23.0
  max_lon: 93.0
  max_lat: 26.5
```

Rules:

- `-180 ≤ longitude ≤ 180`
- `-90 ≤ latitude ≤ 90`
- minimum values must be strictly smaller than maximum values
- antimeridian-crossing rectangles are not supported in version 0.1

The rectangle is used both for Earth Engine clipping and xarray coordinate
subsetting. No polygon shapefile is required.

## 6. CHIRPS mode

Copy `configs/example_chirps.yml` and edit the Cloud project, AOI, models,
future windows, and output directory.

```yaml
precipitation_reference:
  mode: chirps
  chirps_collection: UCSB-CHG/CHIRPS/DAILY
```

CHIRPS is requested only for the AOI and calibration period. Temperature
references always come from ERA5-Land.

## 7. MSWEP mode

### 7.1 Place the data

For Colab, put the authorized file in Drive:

```text
/content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
```

Do not upload the same large file for every job. For repeated arbitrary AOIs,
convert the archive once to a time/latitude/longitude-chunked Zarr store.

### 7.2 Inspect the file

```bash
cloud-qdm inspect-mswep /content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
```

Record the precipitation variable and coordinate names. Check whether the data
are already daily and whether precipitation is an accumulation or rate.

### 7.3 Configure

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

`unit_scale` is multiplied into the values. The final data must be daily
millimetres with units `mm d-1`. If an hourly product stores millimetres per
hourly step, set `aggregate_to_daily: true`. Do not aggregate a rate without
first converting it to an amount per timestep.

Glob patterns such as `/path/mswep_daily_*.nc` are supported through
`xarray.open_mfdataset`.

## 8. Model and period selection

The Earth Engine collection supplies only `ssp245` and `ssp585`. Model IDs are
validated against the documented collection. `GFDL-CM4` contains two grid
configurations and therefore requires a `grid_labels` entry.

Example:

```yaml
models:
  - ACCESS-CM2
  - MIROC6
scenarios:
  - ssp245
  - ssp585
future_windows:
  - start: 2015-01-01
    end: 2040-12-31
    label: near-term
  - start: 2041-01-01
    end: 2070-12-31
    label: mid-century
  - start: 2071-01-01
    end: 2100-12-31
    label: late-century
```

Processing all 2015–2100 values as one ranking population is deliberately not
the default. Window lengths should be broadly comparable with the calibration
sample.

## 9. Validation and execution

Validate without a cloud request:

```bash
cloud-qdm validate my-run.yml
```

Run:

```bash
cloud-qdm run my-run.yml
```

The pipeline fails on invalid data by default. Set
`processing.continue_on_model_error: true` only for exploratory ensembles, and
inspect the manifest for failed models before using any ensemble statistics.

## 10. Processing sequence

For each model:

1. Load the four reference variables.
2. Fetch historical NEX-GDDP data for the calibration dates.
3. Normalize units and coordinate order.
4. Interpolate model data to each reference grid.
5. Retain only common daily timestamps and check coverage.
6. Train monthly QDM and save the adjustment parameters.
7. Correct and save the historical period.
8. Fetch, adjust, and save each future scenario/window.
9. Enforce temperature ordering and non-negative precipitation.
10. Append summary and provenance records.

## 11. Outputs and provenance

Every NetCDF includes the source collection, model, experiment, correction
method, calibration period, creation time, and software version where
available. `run-config.yml` is the normalized configuration and
`run-manifest.json` records successes and failures.

`summary.csv` contains overall descriptive statistics. It is a quality-control
aid, not a substitute for spatial and temporal validation.

## 12. Quality-control checklist

Before accepting results:

- Confirm the historical model and reference have at least 95% common dates.
- Plot observed, raw historical, and corrected historical monthly cycles.
- Compare quantiles and wet-day frequency on a held-out period.
- Inspect boundaries for interpolation NaNs.
- Confirm precipitation is in `mm d-1` before QDM.
- Confirm temperature is in Kelvin during QDM.
- Verify `tasmin ≤ tas ≤ tasmax` after adjustment.
- Check that future changes remain plausible and are not clipped by unit errors.
- Record model-specific missing dates and NEX interpolated-day metadata.

## 13. Troubleshooting

### Earth Engine authentication fails

Confirm the project is registered for Earth Engine and rerun `ee.Authenticate()`.
Do not commit a credential file.

### Empty model collection

Check model spelling, scenario, date limits, and `GFDL-CM4` grid label. Only
`ssp245` and `ssp585` are supported by this Earth Engine collection.

### Colab runs out of memory

Reduce AOI size or model count, use shorter future windows, and lower spatial
chunk sizes. Run one model at a time. Consider a high-memory runtime.

### MSWEP reads slowly

Avoid a single unchunked global NetCDF on mounted Drive. Pre-clip it or convert
it to chunked Zarr. Copying a regional subset to `/content` can improve speed,
but that temporary copy disappears when the runtime stops.

### QDM unit error

Inspect the DataArray `units` attribute. Expected values are `K` for
temperature and `mm d-1` for precipitation.
