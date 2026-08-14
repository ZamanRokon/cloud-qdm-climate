# Methodology and validation

## Quantile Delta Mapping

QDM learns the relationship between an observational reference and a model's
historical distribution, separately for each model, variable, grid cell, and
calendar month. The learned relationship is then applied to historical data
and to future windows.

Temperature uses an additive correction. Precipitation uses a multiplicative
correction with optional frequency adaptation below a configured wet-day
threshold. Fifty quantile nodes are the default.

QDM is selected because it is designed to preserve modeled changes in
quantiles better than ordinary quantile mapping. It remains a statistical
bias-adjustment method and assumes that the calibrated relationship remains
useful under future conditions.

## Reference and model grids

ERA5-Land and MSWEP are approximately 0.1° products, while CHIRPS is 0.05° and
NEX-GDDP-CMIP6 is approximately 0.25°. Model fields are interpolated to the
reference coordinates before training. Finer output coordinates do not imply
new independent dynamical information at that scale.

This implementation preserves missing values rather than filling them with
arbitrary climate constants. Coverage failures are explicit.

## Calibration and adjustment windows

The default 1981–2014 calibration period is common to MSWEP, CHIRPS,
ERA5-Land, and the historical NEX experiment. Future scenarios start in 2015.
Future windows are kept broadly comparable to the training length because QDM
ranks values within the adjustment sample.

## Physical consistency

`tas`, `tasmin`, and `tasmax` are corrected independently. The pipeline then
sorts the three corrected values at every time and grid cell so the output
satisfies `tasmin ≤ tas ≤ tasmax`. This is a pragmatic univariate consistency
step, not a multivariate bias-adjustment model.

Precipitation is clipped at zero after adjustment. Frequency adaptation is
performed by xsdba when enabled.

## Required validation

A mature application must include an out-of-sample evaluation, for example:

- train on 1981–2004 and validate on 2005–2014;
- compare raw and corrected bias in mean, variance, quantiles, extremes, and
  wet-day frequency;
- map performance rather than relying only on AOI means;
- compare multiple reference datasets where possible;
- examine whether modeled climate-change signals are preserved;
- report uncertainty across models and reference products.

The software produces corrected data but cannot establish fitness for a
particular impact study automatically.

## Reproducibility

Archive the normalized run configuration, manifest, package versions, data
access date, model/version properties, and QDM parameter files. Pin a release
tag for published analyses and cite all source datasets.
