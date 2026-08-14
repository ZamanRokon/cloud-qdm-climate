# Methodology and validation

This page defines the implemented statistical method, its assumptions, and the
minimum validation expected before impact analysis. Operating instructions are
in the [technical manual](technical-manual.md).

## 1. Implemented QDM

Quantile Delta Mapping (QDM) is fitted independently for every model, variable,
reference-grid cell, and calendar month with `xsdba.adjustment.QuantileDeltaMapping`.
The default is 50 quantile nodes, linear interpolation, and constant
extrapolation. Temperature uses additive QDM (`kind="+"`); precipitation uses
multiplicative QDM (`kind="*"`).

For a value `x` in an adjustment period, let `p` be its quantile in that
period's model distribution. Let `Q_obs,h(p)` and `Q_mod,h(p)` be historical
reference and model quantiles.

Additive QDM for temperature is:

```text
x_corrected = Q_obs,h(p) + [x - Q_mod,h(p)]
```

Multiplicative QDM for precipitation is:

```text
x_corrected = Q_obs,h(p) * [x / Q_mod,h(p)]
```

Thus the modeled absolute temperature change or relative precipitation change
at a quantile is carried onto the reference distribution. Numerical handling
near zero, quantile interpolation, extrapolation, and optional precipitation
frequency adaptation are delegated to `xsdba`.

The implementation follows the QDM concept described by
[Cannon, Sobie and Murdock (2015)](https://doi.org/10.1175/JCLI-D-14-00754.1).
The exact software behavior is defined by the pinned-compatible `xsdba`
version recorded in the run manifest.

## 2. Data and processing order

| Variable | Reference | Model input | QDM | Output constraint |
|---|---|---|---|---|
| `tas` | ERA5-Land daily mean 2 m temperature | NEX-GDDP-CMIP6 | additive | ordered with min/max |
| `tasmax` | ERA5-Land daily maximum 2 m temperature | NEX-GDDP-CMIP6 | additive | maximum of corrected triplet |
| `tasmin` | ERA5-Land daily minimum 2 m temperature | NEX-GDDP-CMIP6 | additive | minimum of corrected triplet |
| `pr` | CHIRPS or user-supplied MSWEP | NEX-GDDP-CMIP6 | multiplicative | non-finite QDM results at finite inputs set to zero; clipped at zero |

The processing order is fixed:

1. Subset references and model data to the AOI and dates.
2. Normalize coordinates and units (K; mm d-1).
3. Fetch the model with a 0.5-degree buffer, then bilinearly interpolate it to
   the unbuffered reference grid.
4. Align common historical dates and enforce minimum temporal coverage.
5. Train monthly QDM on historical model/reference pairs.
6. Apply the fitted adjustment to historical data and each future window.
7. Enforce variable-domain constraints and write provenance-rich segments.
8. Validate and merge future segments into one daily 2015-2100 NetCDF per
   model, scenario, and variable.

The default calibration is 1981-2014 because it overlaps the configured
references and NEX historical experiment. Future windows must be chronological,
gap-free, and collectively cover 2015-2100. Windows should contain enough
samples per month for stable quantiles and should be stated explicitly in any
publication. They remain analysis/QDM-ranking windows even though the final
NetCDF is continuous.

## 3. Important interpretations

### Grid resolution

NEX-GDDP-CMIP6 is approximately 0.25 degrees, ERA5-Land and MSWEP are roughly
0.1 degrees, and CHIRPS is 0.05 degrees. Interpolation aligns coordinates; it
does not create independent climate dynamics at the finer spacing. Bilinear
interpolation is not mass-conservative for precipitation, so water-balance
applications should quantify the effect or implement a validated conservative
alternative.

The model request is spatially buffered before interpolation. This supplies
source cells on both sides of the target boundary and prevents small AOIs from
acquiring persistent NaN edge rows or columns merely because the model was
clipped too tightly.

### Windows and stationarity

QDM assumes the historical model-reference relationship remains useful in the
future. This repository ranks values separately inside each configured future
window. It does not implement the moving future windows or three-month seasonal
pooling sometimes used in QDM studies. Window choices therefore form part of
the method and must be sensitivity-tested.

### Physical dependence

The four variables are adjusted separately. Sorting corrected `tasmin`, `tas`,
and `tasmax` guarantees `tasmin <= tas <= tasmax`, but it is a pragmatic
constraint rather than multivariate bias adjustment. Univariate QDM does not
guarantee preservation of inter-variable, temporal, or spatial dependence.

### Wet days and tails

When enabled, `xsdba` adapts precipitation frequency below the configured wet-
day threshold. Validate dry-day frequency and light-rain amounts because this
step can affect occurrence statistics. Constant extrapolation applies the
nearest trained correction beyond the outer quantile nodes; it does not
independently estimate extreme-value tails.

Frequency adaptation uses random tie-breaking and trace-value generation.
`qdm.random_seed` makes that step repeatable for each variable, period, and
operation while preserving the calling process's NumPy random state.

For MSWEP, non-finite input values are rejected unless
`fill_non_finite_with_zero` is explicitly enabled. That option is appropriate
only after confirming the values mean no precipitation rather than missing
observations. After adjustment, an otherwise finite model precipitation value
that produces NaN or infinity is set to zero; missing model input remains
missing. Infinity is never written to final NetCDF files.

## 4. Validation design

At minimum, use a held-out historical period. One defensible starting design is
training on 1981-2004 and evaluating 2005-2014, provided data coverage and the
application justify those dates. Do not evaluate only on the training sample.

When `figures.enabled` is true, the pipeline implements this split explicitly:
it fits a temporary QDM using only `evaluation.training`, adjusts only
`evaluation.validation`, and compares reference, raw model, and corrected model
on the held-out dates. It then fits the production QDM on the full calibration
period. Evaluation metrics use cosine-latitude-weighted AOI-mean daily series;
spatial bias panels calculate time-mean fields at every reference-grid cell.

Taylor diagrams report correlation, normalized standard deviation, and
centered RMSE. They do not show mean bias, extremes, wet-day occurrence, or
dependence, so they must be interpreted with the accompanying metric, quantile,
map, and extremes panels. See the [paper-figure guide](paper-figures.md).

Report results by season and location, including:

- mean bias, variability, monthly cycle, and selected quantiles;
- hot/cold extremes, heavy precipitation, wet-day frequency, and spell lengths;
- maps of performance and missing values, not only AOI averages;
- raw versus corrected future absolute temperature and relative precipitation
  changes at several quantiles;
- sensitivity to reference product, calibration dates, quantile count, wet-day
  threshold, regridding, and future-window definition; and
- uncertainty across models, scenarios, and reference datasets.

Inspect boundaries for interpolation gaps and verify units, `pr >= 0`, and
`tasmin <= tas <= tasmax`. Domain-specific impact thresholds and long return
periods require their own evaluation; `summary.csv` cannot establish fitness.

## 5. Reproducibility

Archive the normalized configuration (including `qdm.random_seed`), manifest,
QDM parameter files, logs, package versions, data-access date, collection IDs,
model/grid labels, and validation code. Record the MSWEP version without
exposing a private path or redistributing data. Pin a repository release for
published work and cite every source dataset.
