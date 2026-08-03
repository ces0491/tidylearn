# tidylearn 0.4.0

This is a minor release of a package already on CRAN (0.3.1). It adds a
compute-backend layer (local GPU detection and routing, plus a runtime and
memory advisor), and fixes a group of bugs in model evaluation, pipeline
prediction and AutoML ranking that produced wrong results rather than
errors.

## R CMD check results

0 errors | 0 warnings | 0 notes

A local Windows check intermittently reports one additional NOTE, "unable to
verify current time". This is environmental — it appears only when the check
machine cannot reach the time-verification web service — and is unrelated to
the package.

## Changes in this version

### New features

* `tl_check_gpu()` reports local NVIDIA CUDA support and which GPU-capable
  backends are installed. Detection is cheap and side-effect free: it looks
  for `nvidia-smi` on the PATH and checks installed packages, without
  loading Python or fitting a model. It reports no GPU rather than failing
  on machines without one.
* `tl_compute_advisor()` estimates runtime, peak memory and cost across
  local CPU, local GPU and cloud tiers for a given method and dataset.
* `tl_model()` gains a `compute` argument (`"cpu"`, `"gpu"`, `"auto"`,
  `"cloud"`). The default is `"cpu"`, so existing behaviour is unchanged.
  `"cloud"` is reserved and errors with an explanatory message; no network
  access is attempted anywhere in the package.

### Bug fixes

Four defects returned plausible but incorrect numbers rather than erroring,
so they were not visible to users:

* `tl_evaluate()` scored classification models against raw prediction
  output rather than class labels. Since `predict()` returns probabilities
  for logistic regression, every logistic model scored an accuracy of
  exactly 0. This propagated to `tl_cv()`, `tl_tune_grid()`,
  `tl_run_pipeline()` and `tl_auto_ml()`, all of which ranked logistic
  models last regardless of performance.
* `tl_evaluate()` had no `metrics` argument, so a requested metric was
  silently forwarded to `predict()` and ignored. It now accepts `metrics`
  and computes the requested set.
* `tl_predict_pipeline()` derived its centre and scale from data that had
  already been standardised, so new data reached the model in raw units.
  `tl_run_pipeline()` now records the statistics it learned and prediction
  replays them.
* `tl_auto_ml()` leaderboard scores were always `NA` because the score
  extractor expected a result shape neither evaluation path produces.

Also fixed: prediction for `ridge`/`lasso`/`elastic_net` built a design
matrix that disagreed with the fit whenever a factor predictor was present;
`method = "boost"` could not fit a classification model at all; `plot()`
failed for every unsupervised method; and unsupervised `predict()` used row
count to decide whether new data had been supplied, silently returning
training results for same-sized input.

### Documentation

Vignette examples that printed incorrect results have been corrected — one
reported 0% accuracy in five places, another reported chance-level accuracy
for its complete-workflow example. Added package-level documentation, so
`?tidylearn` resolves.

## Test environments

* local: Windows 11 x64, R 4.5.2
* win-builder: R-devel (2026-07-30 r90327 ucrt) -- Status: OK
* win-builder: R-release (4.6.1, 2026-06-24 ucrt) -- Status: OK

## Downstream dependencies

There are no reverse dependencies.
