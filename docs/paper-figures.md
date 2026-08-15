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

With `figures.enabled: true` and `figures.model_by_model: false`, the pipeline
writes only these all-model products at 600-DPI PNG plus their CSV source data:

| Figure | What it answers | Important interpretation |
|---|---|---|
| `evaluation-taylor-by-variable` | Do models reproduce daily regional variability before and after QDM? | Eight first-quadrant panels show four variables × two stages. Radius is normalized SD; angle is arccos(correlation); dashed contours are normalized centered RMSD. |
| `evaluation-skill-improvement` | How much does QDM reduce normalized RMSE? | Positive percentages are improvement; inspect negative cells rather than hiding them. |
| `projection-ensemble-change` | How do models/scenarios/windows differ? | Boxes summarize model spread, not probabilistic confidence intervals. |

The Taylor construction follows
[Taylor (2001)](https://pcmdi.llnl.gov/report/pdf/55.pdf). QDM change-signal
plots follow the central diagnostic in
[Cannon, Sobie and Murdock (2015)](https://doi.org/10.1175/JCLI-D-14-00754.1).
Before/after panels for a variable share the same radial limit. The reference
point is correlation 1 and normalized SD 1. If a model correlation is negative,
the quarter-circle display places it on the correlation-zero boundary with a
red edge; the exact negative value remains in `evaluation-metrics.csv`.

## 3. Supplementary figures

Set `figures.model_by_model: true` to add these products. This can create many
files and materially increase runtime.

| Figure | Recommended use |
|---|---|
| Per-model `taylor-diagram` | Identify models hidden by an ensemble summary. |
| Per-model `skill-metrics.csv` | Report bias, MAE, RMSE, normalized errors, correlation, SD ratio, and sample size. |
| `seasonal-cycle` | Check monthly climatology; curves are cosine-latitude-weighted AOI means. |
| `distribution-and-quantile-bias` | Check Q-Q proximity to 1:1 and residual central/tail quantile bias. |
| `spatial-bias-<variable>` | Map where QDM helps or worsens bias; precipitation percent bias requires reference mean ≥ 0.1 mm d-1. |
| `wet-day-diagnostics` | Show monthly occurrence and conditional intensity for precipitation. |
| `annual-extremes` | Compare regional PRCPTOT, Rx1day, Rx5day, CDD, TXx, TNn, and DTR; PRCPTOT and CDD use the canonical 1 mm wet-day threshold. |
| `intervariable-dependence` | Disclose changes in pairwise dependence from independent univariate adjustment. |
| `quantile-change-signal` | Check preservation of absolute temperature and relative precipitation change across quantiles. |
| `change-map-<variable>` | Map raw/corrected change and the signal alteration introduced by QDM. |
| `annual-projection-series` | Show within-window annual evolution before and after QDM. |

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

- **Taylor:** “First-quadrant Taylor diagrams of held-out daily AOI-mean
  precipitation and temperature across [N] models for [period], shown before
  and after QDM. Radial distance is standard deviation normalized by the
  reference, angle is arccos(Pearson correlation), and dashed contours show
  normalized centered RMSD.”
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
5. retain the default 600-DPI PNG and export PDF or SVG only when the journal
   requests vector line art;
6. use journal-required fonts, dimensions, accessibility, and color standards;
7. archive the CSV metrics and exact normalized run configuration; and
8. never describe the model ensemble spread as a probability distribution.

The IPCC Interactive Atlas similarly combines maps, time series, annual cycles,
scatter views, and summary tables rather than relying on one plot type; see its
[regional-information guidance](https://interactive-atlas.ipcc.ch/regional-information/guidance).
