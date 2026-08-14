# Technical manual

This is the operating guide. For the scientific basis and validation design,
use [Methodology and validation](methodology.md).

## 1. Runtime and prerequisites

Earth Engine filters ERA5-Land, CHIRPS, and NEX-GDDP-CMIP6. `xee` exposes the
requested subsets to Python; `xarray`, Dask, and `xsdba` perform regridding and
QDM. Google Drive can hold persistent Colab inputs and outputs, but computation
occurs in the Colab runtime.

You need Python 3.10-3.13, an Earth Engine-enabled Google Cloud project, and
enough memory for the AOI. MSWEP mode also requires an authorized NetCDF or
Zarr copy. Never put access tokens or service-account JSON in YAML or Git.

### Google Colab

The supplied [notebook](https://github.com/ZamanRokon/cloud-qdm-climate/blob/main/notebooks/cloud_qdm_colab.ipynb) performs these
steps. The equivalent setup is:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/ZamanRokon/cloud-qdm-climate.git
%cd cloud-qdm-climate
!python -m pip install -e ".[cloud]"

import ee
ee.Authenticate()
ee.Initialize(project="your-earth-engine-project")
```

Keep the runtime connected. Files already written to Drive survive a runtime
reset; files under `/content` do not.

### Linux or WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cloud,test]"
```

## 2. Configure a run

Copy one complete, validated template:

- [`example_chirps.yml`](https://github.com/ZamanRokon/cloud-qdm-climate/blob/main/configs/example_chirps.yml) for cloud-hosted
  precipitation; or
- [`example_mswep.yml`](https://github.com/ZamanRokon/cloud-qdm-climate/blob/main/configs/example_mswep.yml) for user-supplied
  precipitation; or
- [`example_paper.yml`](https://github.com/ZamanRokon/cloud-qdm-climate/blob/main/configs/example_paper.yml) for independent evaluation and paper figures.

Change values in the copy; keep the example files as known-good references.

### Required fields

| Field | Meaning and rule |
|---|---|
| `run.name` | Safe output subdirectory name; no slash |
| `run.output_dir` | Parent output directory, usually on Drive in Colab |
| `earth_engine.project_id` | Earth Engine-enabled Cloud project |
| `aoi.*` | EPSG:4326 rectangle: `min_lon`, `min_lat`, `max_lon`, `max_lat` |
| `calibration` | Historical start/end; end must be on or before 2014-12-31 |
| `future_windows` | Named, continuous processing/analysis windows that collectively cover 2015-2100 |
| `models` | NEX-GDDP-CMIP6 model IDs |
| `scenarios` | `ssp245`, `ssp585`, or both |
| `precipitation_reference.mode` | Exactly `chirps` or `mswep` |

AOI coordinates must satisfy
`-180 <= min_lon < max_lon <= 180` and
`-90 <= min_lat < max_lat <= 90`. Antimeridian-crossing rectangles are not
supported. A polygon or shapefile is not needed.

`GFDL-CM4` has two collection grids. When selected, add:

```yaml
grid_labels:
  GFDL-CM4: gr1  # or gr2
```

### QDM controls

| Field | Default | Operational meaning |
|---|---:|---|
| `nquantiles` | `50` | Quantile nodes; minimum 5 |
| `group` | `time.month` | Calendar-month training; currently the only supported group |
| `wet_day_threshold_mm` | `0.1` | Precipitation frequency threshold in mm d-1 |
| `adapt_wet_day_frequency` | `true` | Enable `xsdba` dry/wet frequency adaptation |
| `random_seed` | `42` | Reproducible wet-day randomization seed |
| `interpolation` | `linear` | Interpolation between trained quantile factors |
| `extrapolation` | `constant` | Use the nearest trained factor outside the quantile range |

Keep these defaults until a validation experiment justifies changing them.

### Processing controls

| Field | Default | Use |
|---|---:|---|
| `earth_engine_chunk_years` | `10` | Years in each multiband Earth Engine request |
| `latitude_chunk` | `40` | MSWEP/Dask latitude chunk |
| `longitude_chunk` | `40` | MSWEP/Dask longitude chunk |
| `netcdf_compression_level` | `1` | Final NetCDF compression, from 0 (off) to 9 (smallest/slowest) |
| `scratch_dir` | unset | Fast local parent directory for disposable future-window files |
| `minimum_time_coverage` | `0.95` | Minimum shared calibration dates |
| `continue_on_model_error` | `false` | Continue an exploratory ensemble after one model fails |
| `save_reference_subsets` | `false` | Persist reference AOI subsets for audit/debugging |

The workflow requests the three ERA5-Land temperature bands together and all
four NEX-GDDP bands together. Reduce `earth_engine_chunk_years` to 5 if Earth
Engine times out for a large AOI; this reduces each request, not QDM memory.
Each monthly training group must span its time samples.

For Colab, set `scratch_dir: /content/cloud-qdm-scratch`. Temporary window
files then use fast runtime storage and no compression; continuous 2015-2100
products are still compressed and written to `run.output_dir`. Scratch files
are disposable and do not survive a runtime reset.

### Independent evaluation and figures

Figures are off by default. Enabling them requires an explicit chronological,
non-overlapping training/validation split inside the full calibration period:

```yaml
evaluation:
  training:
    start: 1981-01-01
    end: 2004-12-31
    label: evaluation-training
  validation:
    start: 2005-01-01
    end: 2014-12-31
    label: independent-evaluation

figures:
  enabled: true
  formats: [png, pdf]  # png, pdf, and svg are supported
  dpi: 300             # 150-600; applies to raster output
```

The evaluation adjustment is trained only on `evaluation.training` and scored
only on `evaluation.validation`. It is not used for projections. After
evaluation, the pipeline trains the saved production adjustment on the complete
`calibration` period. This provides held-out diagnostics without discarding the
validation years from the final fit.

Figure generation deliberately retains the full historical baseline in memory
while future windows are processed and repeats several aggregations. Start with
one model and PNG only. Add PDF after confirming the workflow fits the runtime.

## 3. Prepare MSWEP correctly

MSWEP can be one NetCDF file, a glob such as `/path/mswep_daily_*.nc`, or a
Zarr store. In Colab, store it once on Drive, for example:

```text
/content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
```

Inspect metadata without loading the arrays:

```bash
qdm inspect-mswep /content/drive/MyDrive/cloud-qdm/inputs/mswep_daily.nc
```

Set `variable`, `latitude_name`, `longitude_name`, and `time_name` to the names
reported. `unit_scale` multiplies the values. Data entering QDM must represent
daily precipitation in `mm d-1`.

NaN and infinity are not assumed to be dry days automatically. The default is
to stop and report their count. If inspection of the source documentation and
values confirms that they are dry-day placeholders, set:

```yaml
precipitation_reference:
  mode: mswep
  # other MSWEP fields...
  fill_non_finite_with_zero: true
```

The normalized run configuration and precipitation attributes record the
policy and number replaced. Negative precipitation is always rejected.

Set `aggregate_to_daily: true` only when each source value is an amount for its
timestep. Convert rates to timestep amounts first; summing an unconverted rate
is dimensionally wrong. For repeated large jobs, a time/latitude/longitude-
chunked Zarr store or a pre-clipped regional file is usually faster than a
single global NetCDF on mounted Drive.

Review [Data licensing and governance](data-governance.md) before moving or
sharing MSWEP data.

## 4. Validate and run

Validation reads YAML and checks names, bounds, dates, modes, and numeric
settings without contacting Earth Engine:

```bash
qdm validate my-run.yml
```

Run only after validation succeeds:

```bash
qdm run my-run.yml
```

The pipeline processes models sequentially. When figures are enabled, it first
runs the independent evaluation. It then trains four production QDM
adjustments, saves corrected historical data, and processes every
scenario/window to temporary files. After each scenario, those files are
validated and merged into continuous daily 2015-2100 outputs. Default behavior
stops at the first model failure. If
`continue_on_model_error: true`, always inspect `run-manifest.json` before
calculating ensemble statistics.

For a first test, use one model, one scenario, and a small AOI. Keep the windows
continuous from 2015 through 2100 so the final-file completeness check remains
active. Scale only after checking the complete output and logs. A stopped run
does not automatically skip completed models; use a new `run.name` or remove
only the incomplete run after verifying its path.

### Performance expectations

One configured model is not one calculation: with both scenarios and three
windows it includes historical training for four variables, 24 future
variable-window corrections, temporary writes, four continuous-file merges per
scenario, summaries, and optionally figures. Runtime therefore depends strongly
on AOI size, Earth Engine response time, Drive I/O, and figure settings.

The optimized path batches multiband Earth Engine reads, writes related
variables through one shared Dask graph, computes summaries from the saved
files instead of rerunning QDM, and uses low final-file compression. For a fast
production run:

1. Use `scratch_dir: /content/cloud-qdm-scratch` in Colab.
2. Keep `earth_engine_chunk_years: 10`; fall back to 5 only after a timeout.
3. Use `netcdf_compression_level: 1`; higher levels spend more CPU for smaller files.
4. Leave `figures.enabled: false` during correction, then make paper figures in
   a dedicated analysis run when practical.
5. Keep MSWEP regional and chunked, and avoid reading a global NetCDF directly
   from mounted Drive repeatedly.

## 5. Read the outputs

| Path | Purpose |
|---|---|
| `adjustments/<model>/qdm_<variable>.nc` | Trained factors for provenance/reuse |
| `corrected/<model>/<scenario>/2015-2100/<variable>.nc` | Continuous daily future field on the reference grid |
| `run-config.yml` | Normalized effective configuration |
| `run-manifest.json` | Run/model status, versions, and output paths |
| `summary.csv` | Whole-array descriptive statistics for screening |
| `logs/pipeline.log` | Progress and exception details |
| `references/<variable>.nc` | Optional saved references |
| `figures/core/` | Cross-model paper figures and consolidated CSV metrics |
| `figures/by-model/...` | Model-level evaluation and projection figures |

`summary.csv` is a screening aid, not scientific validation.
The [paper-figure guide](paper-figures.md) defines every plotted statistic and
explains which figures belong in the main text or supplement.

Before accepting a run:

1. Confirm every intended model is `complete` in the manifest.
2. Check units: K for temperature and mm d-1 for precipitation.
3. Check `finite_fraction` and `non_finite_values` in `summary.csv`, then map
   any remaining missing values.
4. Verify `tasmin <= tas <= tasmax` and `pr >= 0`.
5. Compare raw, reference, and corrected historical distributions on held-out
   dates, including extremes and wet-day frequency.
6. Test future change-signal preservation and report ensemble/reference
   uncertainty.

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| Earth Engine initialization fails | Authenticate again; verify the project is Earth Engine enabled and matches `project_id` |
| No model images | Check exact model ID, scenario, dates, and the `GFDL-CM4` grid label |
| MSWEP variable missing | Run `inspect-mswep`; copy exact variable and coordinate names |
| MSWEP selection is empty | Check coordinate order/range, AOI overlap, and calibration dates |
| MSWEP non-finite error | Verify whether the values truly mean dry days; only then enable `fill_non_finite_with_zero` |
| Colab memory failure | Reduce AOI/model count; shorten windows; process one model per run; consider high-memory compute |
| MSWEP is slow on Drive | Pre-clip, rechunk to Zarr, or copy an authorized regional subset to `/content` temporarily |
| QDM unit error | Confirm source meaning and units before applying `unit_scale` or daily aggregation |
| Figure generation is slow | Use one model, one window, PNG only, and a smaller AOI for the test run |
| Figure validation fails | Confirm training ends before validation starts and both periods lie inside `calibration` |
| Future merge is incomplete | Confirm windows continuously cover 2015-01-01 through 2100-12-31 and inspect the failed segment log |

When reporting a failure, include the command, sanitized YAML, final log lines,
Python version, and package version. Never include credentials or a licensed
data file.
