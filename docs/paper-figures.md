# Paper-figure guide

The figure suite is broad, but a paper should use only figures that answer its
research questions. A typical main text needs five to eight figures; place
supporting diagnostics in supplementary material. Every plotted value is also
written to CSV so results can be checked independently.

## 1. Scientifically valid setup

Use [`example_paper.yml`](https://github.com/ZamanRokon/cloud-qdm-climate/blob/main/configs/example_paper.yml).
Its full 1981-2014 calibration is used for the production fit. A separate QDM
is trained on 1981-2004 and evaluated on held-out 2005-2014 data. Adapt these
dates to data availability and the study design; keep validation later than
training and do not let the periods overlap.

A plot from the full calibration output is an in-sample fit diagnostic, not
independent evidence of performance. The automated paper figures use the
held-out evaluation data for that reason. This follows the principle that
downscaling validation must assess application-relevant distributions,
temporal behavior, spatial behavior, extremes, and climate-change response
rather than one aggregate score ([VALUE framework](https://doi.org/10.1002/2014EF000259)).

## 2. Core figures

| Figure | What it answers | Important interpretation |
|---|---|---|
| `evaluation-taylor-by-variable` | Do models reproduce daily regional variability? | Radius is normalized SD; angle is correlation; gray contours are centered RMSE. Mean bias is absent. |
| `evaluation-skill-improvement` | How much does QDM reduce normalized RMSE? | Positive percentages are improvement; inspect negative cells rather than hiding them. |
| `seasonal-cycle` | Does QDM improve monthly climatology? | Curves are cosine-latitude-weighted AOI means. |
| `distribution-and-quantile-bias` | Are central and tail quantiles corrected? | Q-Q proximity to 1:1 and residual quantile bias must both be shown. |
| `spatial-bias-<variable>` | Where does QDM help or worsen bias? | Precipitation uses percent bias where reference mean is at least 0.1 mm d-1. |
| `quantile-change-signal` | Are modeled future changes retained across quantiles? | Temperature change is absolute; precipitation change is relative. Equality is desirable but not proof the raw signal is realistic. |
| `change-map-<variable>` | Where do raw and corrected changes differ? | The third panel isolates the signal alteration introduced by QDM. |
| `projection-ensemble-change` | How do models/scenarios/windows differ? | Boxes summarize model spread, not probabilistic confidence intervals. |

The Taylor construction follows
[Taylor (2001)](https://pcmdi.llnl.gov/report/pdf/55.pdf). QDM change-signal
plots follow the central diagnostic in
[Cannon, Sobie and Murdock (2015)](https://doi.org/10.1175/JCLI-D-14-00754.1).

## 3. Supplementary figures

| Figure | Recommended use |
|---|---|
| `wet-day-diagnostics` | Show monthly occurrence and conditional intensity for precipitation. |
| `annual-extremes` | Compare regional PRCPTOT, Rx1day, Rx5day, CDD, TXx, TNn, and DTR; PRCPTOT and CDD use the canonical 1 mm wet-day threshold. |
| `intervariable-dependence` | Disclose changes in pairwise dependence from independent univariate adjustment. |
| `annual-projection-series` | Show within-window annual evolution before and after QDM. |
| Per-model `taylor-diagram` | Identify models hidden by an ensemble summary. |
| Per-model `skill-metrics.csv` | Report bias, MAE, RMSE, normalized errors, correlation, SD ratio, and sample size. |

The extremes are regional, AOI-mean, ETCCDI-style diagnostics. They are not a
replacement for grid-cell ETCCDI indices, station extremes, threshold-specific
impact indices, or a formal extreme-value/return-level analysis. The broader
index definitions are available from the
[ETCCDI](https://etccdi.pacificclimate.org/indices_def.shtml) and implemented
extensively by [xclim](https://xclim.readthedocs.io/en/stable/genindex.html).
The configurable QDM drizzle threshold is separate from the fixed 1 mm ETCCDI
threshold. `qdm.random_seed` makes `xsdba` wet-day randomization reproducible.

## 4. Suggested caption starters

Adapt these; do not copy them without inserting the AOI, dates, reference,
models, sample size, and weighting.

- **Taylor:** “Taylor diagram of held-out daily AOI-mean [variable] for
  [period]. Radial distance is standard deviation normalized by [reference],
  angle is Pearson correlation, and contours show normalized centered RMSE.”
- **Spatial bias:** “Held-out climatological [variable] reference, raw-model
  bias, QDM-corrected bias, and reduction in absolute bias for [period].”
- **Quantiles:** “Reference-versus-model quantiles and residual quantile bias
  for raw and QDM-corrected daily AOI means during independent evaluation.”
- **Change signal:** “Raw and QDM-corrected changes in historical-to-[window]
  quantiles under [scenario]; temperature changes are in degrees Celsius and
  precipitation changes are percentages.”
- **Ensemble:** “Distribution across [N] models of median-quantile change by
  scenario and future window; boxes represent inter-model spread.”

## 5. What is not automated

The suite does not choose statistically significant regions, add stippling,
calculate confidence intervals, test field significance, fit return periods,
validate against stations, draw administrative boundaries, or correct spatial
and inter-variable dependence. Those require study-specific hypotheses,
sampling assumptions, external geography, or additional data. Univariate QDM
can improve marginal distributions while leaving or changing dependence; see
[Cannon (2016)](https://doi.org/10.1175/JCLI-D-15-0679.1).

Before submission, also:

1. state whether diagnostics use grid cells or AOI means;
2. report missing-data rules, calendar handling, wet-day threshold, and units;
3. compare CHIRPS and MSWEP when reference uncertainty matters;
4. test sensitivity to the evaluation split and future-window definitions;
5. export PDF or SVG for line art and retain 300-600 dpi PNG for raster review;
6. use journal-required fonts, dimensions, accessibility, and color standards;
7. archive the CSV metrics and exact normalized run configuration; and
8. never describe the model ensemble spread as a probability distribution.

The IPCC Interactive Atlas similarly combines maps, time series, annual cycles,
scatter views, and summary tables rather than relying on one plot type; see its
[regional-information guidance](https://interactive-atlas.ipcc.ch/regional-information/guidance).
