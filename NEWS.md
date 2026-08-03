# tidylearn 0.4.0

## New Features

### Compute backends (foundation)

* `tl_check_gpu()` — detects local NVIDIA CUDA support and reports which
  GPU-capable backends (xgboost, keras, tensorflow, torch) are
  installed. Cheap detection: parses `nvidia-smi` output and checks
  installed packages without loading Python or fitting a model. Returns
  a `tidylearn_gpu_check` object with a `print()` method.

* `tl_compute_advisor()` — S3 generic that estimates runtime, peak RAM,
  and cost across local CPU, local GPU, and cloud GPU tiers for a given
  tidylearn method and dataset. Dispatches on either a method name
  (`character`) or a fitted `tidylearn_supervised` model. Returns a
  structured recommendation with a `print()` method. Cloud-tier
  estimates are reported but not yet executable; Modal integration will
  follow in a later iteration.

### Compute backends (local GPU routing)

* `tl_model()` now accepts a `compute` argument on both supervised and
  unsupervised paths: `"cpu"` (default — existing behaviour), `"gpu"`
  (route to local CUDA when the method supports it), `"auto"` (consult
  `tl_compute_advisor()` and pick per call), or `"cloud"` (reserved;
  errors with a clear message until the Modal integration lands).

* `tl_fit_xgboost(compute = "gpu")` passes `device = "cuda"` to
  `xgb.train()`. Requires xgboost compiled with CUDA support.

* `tl_fit_deep(compute = "gpu")` defers to TensorFlow's automatic CUDA
  detection — the argument is accepted for API consistency but does not
  itself change the keras model setup.

* All compute validation flows through `tl_resolve_compute()` so the
  behaviour is uniform across paradigms: methods without an upstream
  GPU path (linear, glm, randomForest, pca, kmeans, etc.) warn and fall
  back to CPU when `"gpu"` is requested; `"cloud"` errors the same way
  on supervised and unsupervised methods. The resolved tier is recorded
  on `model$spec$compute` for both paradigms.

### Compute backends (cloud reframed as memory-headroom tier)

* `tl_compute_advisor()` now treats cloud as a "doesn't fit on my
  machine" tier rather than a GPU-acceleration-only tier. Cloud
  estimates are produced for every method the advisor supports (not
  just GPU-eligible ones), and the recommendation flips to `"cloud"`
  whenever the local job is RAM-infeasible — including CPU-only
  methods like linear regression, SVM or random forest on very large
  data.

  Scope: the advisor covers the 13 supervised methods in
  `.tl_method_profiles`. Unsupervised methods (PCA, k-means, MDS,
  clustering) are not modelled and calling the advisor on one errors.
  Reaching the cloud recommendation through `tl_model(compute =
  "auto")` additionally requires a method with an upstream GPU path
  (`xgboost`, `deep`), since `tl_resolve_compute()` short-circuits
  CPU-only methods to `"cpu"` before consulting the advisor. Call
  `tl_compute_advisor()` directly to get memory-headroom advice for the
  other supervised methods.

* New internal Modal instance tier table (`.tl_modal_tiers`) listing
  CPU-RAM tiers (`cpu-small`, `cpu-large`, `cpu-xlarge`) alongside GPU
  tiers (`t4`, `a10g`, `a100-40gb`, `a100-80gb`). The advisor picks the
  cheapest viable tier for the workload based on RAM headroom and
  whether the method has an upstream GPU path. Pricing is approximate
  as of early 2026 and may drift; revise if Modal pricing changes.

* The advisor's recommendation is no longer gated on
  `cloud$configured`. The advisor advises optimally; the caller
  (`tl_resolve_compute()`) decides whether it can act on a cloud
  recommendation. When `compute = "auto"` and the advisor recommends
  cloud, `tl_resolve_compute()` emits a clear message that cloud isn't
  yet wired up and falls back to local CPU.

* Print method updated: the cloud line now shows the chosen tier label
  (e.g., `T4 (16 GB VRAM / 16 GB RAM)`) alongside the time and cost
  estimate.

### Compute backends (security threat model)

* Added `inst/security/threat-model.md` — the contract for what
  cloud compute in tidylearn will and will not do once the Modal
  integration lands. Covers token handling (never read in R), data
  egress consent (per-call `confirm_upload = TRUE` plus session-level
  `tl_cloud_consent()`), ephemeral compute (no persistent Modal
  volumes by default), no telemetry, and an audit checklist that
  reviewers can grep / verify against the Modal-integration PR. The
  doc is shipped with the package so users (and CRAN reviewers) can
  find it via `system.file("security/threat-model.md", package =
  "tidylearn")`.

## Bug Fixes

These four defects produced plausible but wrong numbers rather than
errors, so results computed with earlier versions should be rechecked.

* `tl_evaluate()` scored classification models against raw prediction
  output rather than class labels. Because the default `predict()` type
  returns probabilities for logistic regression, comparing them to
  factor labels gave an accuracy of exactly 0 for every logistic model.
  Evaluation now requests `type = "class"` explicitly. Everything built
  on `tl_evaluate()` was affected — `tl_cv()`, `tl_tune_grid()`,
  `tl_tune_random()`, `tl_run_pipeline()`, `tl_auto_ml()` and
  `tl_compare_cv()` all ranked logistic models last regardless of how
  they actually performed.

* `tl_evaluate()` had no `metrics` argument, so a requested metric
  silently landed in `...` and was forwarded to `predict()`. Only
  accuracy (classification) or rmse/mae/rsq (regression) were ever
  returned. `tl_evaluate()` now takes `metrics` and computes the
  requested set, delegating to `tl_calc_classification_metrics()` for
  classification. Classification supports accuracy, precision, recall,
  sensitivity, specificity, f1, auc and pr_auc; regression supports
  rmse, mse, mae, mape and rsq. `tl_cv()` gains a matching `metrics`
  argument. This removes the "Could not determine best model ... all
  values NA" warning from default pipeline runs and the
  `replacement has length zero` error from
  `tl_tune_grid(metric = "f1")`.

  Regression `rsq` is now `1 - SS_res/SS_tot` rather than the squared
  correlation. The two agree for in-sample OLS; the squared correlation
  was optimistic on held-out data.

* `tl_predict_pipeline()` derived its centre and scale from
  `results$processed_data`, which is stored *after* standardization —
  so new data was rescaled against a mean of ~0 and an sd of ~1 and
  reached the model in raw units. On `mtcars` with `mpg ~ wt + hp` this
  returned predictions near -230 for rows whose actual mpg was 21. The
  same defect made imputation substitute a standardized median (~0) for
  missing values instead of the raw-scale one. `tl_run_pipeline()` now
  records the medians, modes, centres and scales it learned in
  `results$preprocessing_stats`, and `tl_predict_pipeline()` applies
  those. Pipelines run by an earlier version carry no such statistics
  and now raise a clear error asking for a re-run rather than silently
  producing wrong predictions. Constant columns are centred without
  dividing by zero.

* `tl_auto_ml()`'s leaderboard scores were always `NA`.
  `create_leaderboard()` expected a result shape that neither
  `tl_cv()` nor `tl_evaluate()` produces, so every model scored `NA`
  and the reported "best model" was whichever trained first. Score
  extraction now handles both shapes, and the target metric is passed
  through to every evaluation.

* `predict()` on unsupervised models used `nrow(new_data) ==
  nrow(object$data)` to decide whether new data had been supplied. Any
  new data with the same number of rows as the training set silently
  got the training result back — verified with a PCA projection of an
  all-999 frame returning the training scores. `predict()` now tracks
  whether the caller supplied `new_data` rather than inferring it from
  row count. This also affected `predict.tidylearn_transfer()` and
  `predict.tidylearn_stratified()`, which delegate to it.

  Methods with no out-of-sample projection (PAM, CLARA, MDS, DBSCAN,
  hierarchical clustering) now error when handed new data instead of
  returning training assignments that look like predictions. PAM and
  CLARA gained the training-data branch they previously lacked, and
  hierarchical clustering — whose fit holds a tree, not assignments —
  points at `tidy_cutree()` rather than returning `NULL`.

* Prediction for `ridge`, `lasso` and `elastic_net` built its design
  matrix from a `~ predictors - 1` formula while the fit used
  `model.matrix()` with the intercept dropped. The two disagree
  whenever a factor predictor is present: the fit uses treatment
  contrasts (k-1 columns), prediction one-hot encodes (k columns), so
  any such model failed with `The number of variables in newx must be
  N`. The fit now records its terms and factor levels, and prediction
  rebuilds an identically-coded design matrix from them.

* Regularized classification ignored the `type` argument and always
  returned class labels, so `type = "prob"` gave labels and ROC,
  calibration, lift and gain plots could not work for these models.
  `type = "prob"` now returns one probability column per class (binary
  and multinomial), and `type = "class"`/`"response"` returns a factor
  carrying the training levels rather than a character vector. An
  unrecognised type errors instead of silently returning labels.

* `method = "boost"` could not fit a classification model at all:
  `gbm()` was handed a factor response with `distribution =
  "bernoulli"`, which requires a numeric 0/1 response. The response is
  now encoded with the second factor level as the positive class,
  matching the orientation `tl_predict_boost()` already assumed.

* `plot()` failed for every unsupervised method. The `tl_fit_*`
  wrappers unpack the `tidy_*` objects into plain lists, but the plot
  helpers were handed the unpacked list: k-means, PAM, CLARA and
  DBSCAN partial-matched `$cluster` to the `$clusters` tibble and built
  a nested column; PCA and MDS hit `tidy_pca`/`tidy_mds` class checks
  that a plain list cannot satisfy; hclust passed a list where an
  `hclust` object was expected. Each method now supplies the structure
  its plot helper expects.

### Compute backends (corrections)

* `parallel` is now declared in Imports. `tl_estimate_local_cpu_internal()`
  calls `parallel::detectCores()`, which without the declaration produces
  an "'::' call not declared from" NOTE under `R CMD check`.

* `testthat` minimum raised to 3.1.7. The compute tests use
  `local_mocked_bindings()` (3.1.7) and `expect_no_warning()` (3.1.5);
  on an older testthat the suite errored rather than skipped.

* `tl_detect_cuda_internal()` now checks the exit status of `nvidia-smi`.
  A machine with the binary installed but the driver unloaded prints its
  error message to stdout and exits non-zero — that text was being parsed
  as a device name, so `tl_check_gpu()` reported a working GPU and
  `compute = "gpu"` routed `device = "cuda"` into a fit that then failed.

* GPU routing for xgboost now requires xgboost >= 2.0.0, checked during
  backend detection. The `device` parameter arrived in 2.0.0; older
  versions ignore unknown parameters, so the fit ran on CPU while
  `spec$compute` recorded `"gpu"`. Older versions are now reported as
  having no GPU path, so `compute = "gpu"` warns and falls back honestly.

* `tl_model(compute = "auto")` now forwards the caller's runtime-relevant
  hyperparameters to the advisor. Previously the advisor always estimated
  a default-sized job, so `tl_model(..., method = "xgboost", nrounds =
  5000, compute = "auto")` was costed as `nrounds = 100` and could choose
  CPU when GPU was the right call.

* `tl_compute_advisor()` no longer skips a local GPU that finishes
  quickly. The guard required an estimated GPU runtime of at least 5
  seconds on top of a 3x speedup, so a job estimated at 70s on CPU and
  4.7s on GPU — a 15x speedup — was reported as "No meaningfully faster
  tier available". The sub-60s check earlier in the same function
  already covers jobs too small to bother offloading.

* `tl_compute_advisor(fitted_model, formula = ...)` no longer errors with
  "formal argument 'formula' matched by multiple actual arguments". The
  documentation says `formula` is ignored for a fitted model; now it
  actually is.

## Other Changes

* `tl_auto_ml()` now cross-validates the PCA-augmented and
  cluster-augmented variants when the budget allows. Previously these
  were scored on training data while baselines were cross-validated, so
  once scoring worked at all, overfit variants would have outranked
  honestly-scored models. The leaderboard gains an `evaluation` column
  recording `"cv"` or `"train"` per model, since mixed scores are not
  directly comparable.

* `tl_auto_ml()` no longer fits logistic regression to a multiclass
  response — the implementation is binary-only, and the resulting model
  was meaningless. It errors early when the response has fewer than two
  observed classes.

* `tl_run_pipeline()` rejects an unnamed `models` argument. Passing a
  character vector previously trained nothing and failed later with an
  indexing error.

* `tl_evaluate()` errors when the response column is absent from
  `new_data` instead of computing metrics against `NULL`.

* `tl_tune_grid()` and `tl_tune_random()` failed with "argument is of
  length zero" whenever a `metric` was named without also naming
  `maximize`. The optimisation direction was only assigned inside the
  branch that supplies a default metric, so an explicit metric left
  `maximize` at `NULL` and the later `if (maximize)` errored. Direction
  now follows the metric itself: `rmse`, `mse`, `mae` and `mape` are
  minimised, everything else maximised. An explicitly supplied
  `maximize` is still respected.

* Tuning a single hyperparameter dropped its name. Indexing one column
  of the results without `drop = FALSE` collapsed the row to a bare
  value, so the winning setting was passed to `tl_model()` positionally
  and never reached the underlying fit — a tuned `cp` or `lambda` was
  silently discarded. Affected both `tl_tune_grid()` and
  `tl_tune_random()`.

* `tl_plot_tuning_results(plot_type = "importance")` errored on
  categorical parameters with "Can't subset `.data` outside of a data
  mask context". The ANOVA branch built its formula with the tidy-eval
  `.data` pronoun, which `aov()` cannot evaluate; it now uses
  `stats::reformulate()`.

* `tl_plot_tuning_results(plot_type = "grid")` errored with "object 'p'
  not found" when a parameter had more than 20 unique values. The
  fallback to a scatter plot called the function recursively but
  discarded the result.

## Tests

* New `test-metrics.R` and `test-pipeline.R` cover the four fixes
  above; `tl_evaluate()` and the whole pipeline family previously had
  no test coverage, which is why the defects survived. Added
  leaderboard scoring and ranking tests to `test-workflows.R`.

* `tl_auto_ml handles small datasets` used `iris[1:30, ]`, which is
  entirely setosa. It passed only because a degenerate single-class
  logistic model was counted as a trained model. It now samples across
  all three species, and a separate test covers the single-class
  rejection.

* New `test-supervised-predict.R` and `test-unsupervised-predict.R`
  cover the prediction fixes above, and `tests/testthat/setup.R` draws
  base-graphics test plots to a null device so they no longer leave an
  `Rplots.pdf` behind.

## Documentation

* Corrected vignette examples that printed wrong results. The
  integration-workflows vignette reported 0% accuracy in five places —
  it compared logistic regression's probability output against factor
  labels, on a three-class response that logistic regression cannot
  represent. The supervised-learning vignette reported 33.3% (chance)
  for its complete-workflow example, which fitted on standardized
  features and then predicted on raw test data. Both now use
  multiclass-capable methods, score through `tl_evaluate()`, and apply
  the training preprocessing to the test set.

* The getting-started and supervised-learning vignettes now explain
  that `predict()`'s default `type = "response"` returns probabilities
  for logistic regression but class labels for trees and forests, and
  show `type = "class"` and `type = "prob"` alongside `tl_evaluate()`.

* Re-enabled seven vignette chunks that were disabled while the
  underlying bugs were present: ridge, lasso, elastic net and SVM in
  the supervised-learning vignette, and PAM, DBSCAN and CLARA in the
  unsupervised-learning vignette.

* Added package-level documentation, so `?tidylearn` now resolves.

* README: fixed a `predict()` example that referenced columns which do
  not exist, replaced a `plot_clusters()` call that passed a model
  where a data frame is required, and added a section on the compute
  backends.

* `tl_run_pipeline()` documents the `$preprocessing_stats` component,
  and `predict()` no longer advertises unsupervised `type` values that
  it ignores — its `@return` now describes the shape unsupervised
  models actually produce, and which of them accept `new_data`.

* `tl_check_gpu()` and `tl_compute_advisor()` examples now run rather
  than sitting in `\dontrun{}`; neither requires a GPU.

# tidylearn 0.3.1

## Performance

* `tidy_gower()` — eliminated two layers of redundant work in the pairwise
  distance loop:
  * Column ranges (`max - min`) and ordinal rank vectors were previously
    recomputed on every `(i, j)` pair. They are now computed once in a
    pre-pass, reducing work from O(n² × p) to O(n² + p).
  * Replaced scalar data-frame indexing `data[i, k]` — which dispatches to
    the R-level `[.data.frame` method on every call — with pre-extracted
    plain-vector access `col_vecs[[k]][i]`, which resolves at the C level.
    Benchmarks show 10–100× faster scalar access; the gain compounds across
    the full `n*(n-1)/2 * p` iterations.
  * Column types (`is.numeric`, `is.ordered`) are now resolved once into a
    `col_type` character vector, removing repeated S3 predicate calls from
    the inner loop.

## Bug Fixes

* Fixed `tl_reduce_dimensions()` returning the internal `.obs_id` row
  identifier as a column of its `$data` result. Passing that data to a
  supervised model via a `response ~ .` formula fed `.obs_id` in as a
  high-cardinality predictor, which made tree-based fits effectively
  non-terminating. The identifier is now dropped from the returned data,
  consistent with how the pipeline and transfer-learning paths already
  handle it.
* Fixed `print()` and `summary()` erroring on the model objects returned
  by `tl_step_selection()` and `tl_tune_xgboost()`. Both constructed their
  object without the `spec$paradigm` field or the `tidylearn_supervised`
  class, so the print method hit a zero-length `if` condition and
  `summary()` took the unsupervised branch. Both objects are now built
  consistently with `tl_model()`.
* Fixed `tidy_gower()` (and `tidy_dist(..., method = "gower")`) erroring on
  single-row input. The pairwise loop used `1:(n - 1)`, which produces the
  invalid sequence `1:0` when `n` is 1; it now uses `seq_len(n - 1)`, so a
  single-row data frame returns an empty `dist` object, consistent with
  `stats::dist()`.

## Tests

* Added 11 tests for `tidy_gower()` / `tidy_dist(..., method = "gower")`
  covering: return type and metadata, symmetry and self-distance, identical
  rows, hand-verified numeric / categorical / ordered / mixed-type distances,
  NA skipping, custom weights, constant-column denominator behaviour, and
  single-row input.

## Internal

* Removed seven unused packages from `Suggests` (caret, mclust, onnx,
  parsnip, recipes, reticulate, workflows) — none were referenced in
  package code, tests, or vignettes.


# tidylearn 0.3.0

## New Features

### Data Ingestion (`tl_read()` Family)

* New `tl_read()` dispatcher function — auto-detects format from file
  extension, URL pattern, or connection string and routes to the appropriate
  reader
* All readers return a `tidylearn_data` object, a tibble subclass carrying
  source, format, and timestamp metadata via `print.tidylearn_data()`

#### File Format Readers

* `tl_read_csv()` / `tl_read_tsv()` — via readr with base R fallback
* `tl_read_excel()` — `.xls`, `.xlsx`, `.xlsm` files via readxl
* `tl_read_parquet()` — via nanoparquet
* `tl_read_json()` — tabular JSON via jsonlite
* `tl_read_rds()` / `tl_read_rdata()` — native R formats via base R

#### Database Readers

* `tl_read_db()` — query any live DBI connection
* `tl_read_sqlite()` — auto-connect to SQLite files via RSQLite
* `tl_read_postgres()` — connection string or named params via RPostgres
* `tl_read_mysql()` — connection string or named params via RMariaDB
* `tl_read_bigquery()` — Google BigQuery via bigrquery

#### Cloud/API Readers

* `tl_read_s3()` — download and read from S3 URIs via paws.storage
* `tl_read_github()` — download raw files from GitHub repositories
* `tl_read_kaggle()` — download datasets via the Kaggle CLI

#### Multi-File Reading

* `tl_read()` accepts a character vector of paths — reads each and row-binds
  with a `source_file` column
* `tl_read_dir()` — scan a directory for data files with optional format,
  pattern, and recursive filtering
* `tl_read_zip()` — extract and read from zip archives, with optional file
  selection
* All backend packages are suggested dependencies, checked at call time via
  `tl_check_packages()`

### New Vignette

* Added "Data Ingestion with tidylearn" vignette covering all readers,
  databases, cloud sources, multi-file reading, and the full pipeline
* Updated "Getting Started" vignette to include `tl_read()` in the workflow

## Bug Fixes

### Workflow and Pipeline Fixes

* Fixed `tl_transfer_learning()` hanging indefinitely when used with PCA
  pre-training. The `.obs_id` row-identifier column from PCA output was
  being included in the supervised formula, creating a massive dummy-variable
  matrix. The column is now stripped before both training and prediction.
* Fixed `tl_run_pipeline()` failing with "attempt to select less than one
  element" when all cross-validation metrics were NA. Root cause: `scale()`
  returned matrix columns instead of vectors, causing downstream metric
  computation to produce NaN. Added `as.vector()` wrapper and hardened the
  best-model selection to handle all-NA metric values gracefully.
* Overhauled `tl_auto_ml()` time budget enforcement. The budget now controls
  which models are attempted: budgets under 30s skip slow C-level models
  (forest, SVM, XGBoost) entirely, and cross-validation is skipped when
  remaining time is tight. Baseline model order changed to fast-first
  (tree, logistic/linear, then forest). See `?tl_auto_ml` for full details
  on budget tiers.

### Interaction and Prediction Fixes

* Fixed `tl_interaction_effects()` crashing with "unused argument (se.fit)"
  because tidylearn's `predict()` method does not support `se.fit`. Now uses
  `stats::predict()` on the raw model object for confidence intervals. Also
  fixed an invalid formula in the internal slope calculation.
* Fixed `tl_plot_interaction()` expecting `fit`/`lwr`/`upr` columns from
  `predict()` output. Now correctly handles tidylearn's `.pred` tibble
  format.

### Visualization Fixes

* Fixed `tl_plot_intervals()` calling non-existent `tl_prediction_intervals()`
  function. Now computes confidence and prediction intervals directly via
  `stats::predict(..., interval = "confidence")` and
  `stats::predict(..., interval = "prediction")`.
* Fixed `tl_plot_svm_boundary()` erroring with "at least two predictor
  variables required" when using `response ~ .` formulas. The function now
  resolves predictors from data column names instead of `all.vars()`, which
  does not expand `.`. Also switched from `geom_contour_filled` (which
  failed on discrete class predictions) to `geom_raster`.
* Fixed `tl_plot_svm_tuning()` passing `NULL` entries in the `ranges` list
  to `e1071::tune()`, which caused "NA/NaN/Inf in foreign function call"
  errors. Tuning ranges are now built conditionally based on the kernel type.
* Fixed `tl_plot_xgboost_shap_summary()` failing with "arguments imply
  differing number of rows" when `n_samples` differed from `nrow(data)`.
  Sampling is now performed before SHAP computation so that feature values
  and SHAP values always have the same number of rows.

### Other Fixes

* Fixed classification auto-detection silently treating numeric responses
  with <= 10 unique values as classification. The response must now be a
  factor or character for classification; a helpful message is emitted when
  a low-cardinality numeric response is detected.
* Fixed `tl_check_assumptions()` crashing with "list object cannot be
  coerced to logical" when some assumption checks returned NULL (e.g.,
  when optional test packages were not installed).
* Fixed SVM default `gamma` calculation to use predictor count only
  (`1 / (ncol(data) - 1)`) instead of including the response column.
* Added missing `@return` tag to `print.tidylearn_data()`.
* Replaced deprecated ggplot2 `size` parameter with `linewidth` in all
  `geom_line()` calls across visualization, classification, PCA, DBSCAN,
  and validation plotting functions.

## Tests

* Added test suite for visualization module (26 tests) — plot dispatch,
  regression/classification plots, lift/gain charts, model comparison,
  unsupervised visualization, and Shiny dashboard.
* Added test suite for tuning module (49 tests) — `tl_default_param_grid`,
  `tl_tune_grid`, `tl_tune_random`, `tl_plot_tuning_results`, and input
  validation.
* Added test suite for diagnostics module (75 tests) — influence measures,
  influence plots, assumption checking, and outlier detection across all
  methods (IQR, z-score, Cook's, Mahalanobis).

## Code Quality

* Package-wide lint cleanup — all R source files, tests, and vignettes
  now pass lintr with zero issues
* Replaced unsafe `1:n` patterns with `seq_len()` / `seq_along()`
* Removed unused variables across the codebase
* Renamed non-snake_case variables to follow R conventions
* Added `.lintr` configuration enforcing `%>%` pipe consistency

# tidylearn 0.2.0

## New Features

### Formatted gt Tables

* New `tl_table()` dispatcher function — mirrors `plot()` but produces
  formatted `gt` tables instead of ggplot2 visualisations
* `tl_table_metrics()` — styled evaluation metrics table from `tl_evaluate()`
* `tl_table_coefficients()` — model coefficients with p-values (lm/glm) or
  sorted by magnitude (glmnet), with conditional highlighting
* `tl_table_confusion()` — confusion matrix with correct predictions
  highlighted on the diagonal
* `tl_table_importance()` — ranked feature importance with colour gradient
* `tl_table_variance()` — PCA variance explained with cumulative % coloured
* `tl_table_loadings()` — PCA loadings with diverging red–blue colour scale
* `tl_table_clusters()` — cluster sizes and mean feature values for kmeans,
  pam, clara, dbscan, and hclust models
* `tl_table_comparison()` — side-by-side multi-model comparison table
* All table functions share a consistent `gt` theme via internal
  `tl_gt_theme()` helper
* `gt` is a suggested dependency — functions error with an install message if
  `gt` is not available

### New Vignette

* Added "Reporting with tidylearn" vignette covering all plot and table
  functions

## Bug Fixes

* Fixed `tl_fit_dbscan()` returning a non-existent `core_points` field
  instead of `summary` from the underlying `tidy_dbscan()` result

# tidylearn 0.1.1

## Bug Fixes

* Fixed `plot()` failing on supervised models with
  "could not find function 'tl_plot_model'" by implementing the missing
  `tl_plot_model()` and `tl_plot_unsupervised()` internal dispatchers
  ([#1](https://github.com/ces0491/tidylearn/issues/1))
* Fixed `tl_plot_actual_predicted()`, `tl_plot_residuals()`, and
  `tl_plot_confusion()` failing due to accessing a non-existent `$prediction`
  column on predict output (correct column is `$.pred`)
* Fixed the same `$prediction` column mismatch in the `tl_dashboard()`
  predictions table

# tidylearn 0.1.0

## Initial CRAN Release

* First release of tidylearn - a unified tidy interface to R's machine learning
  ecosystem

### Features

#### Unified Interface

* `tl_model()` - Single function to fit 20+ machine learning models
* Consistent function signatures across all methods
* Tidy tibble output for all results
* Access raw model objects via `$fit` for package-specific functionality

#### Supervised Learning Methods

* Linear regression (stats::lm)
* Polynomial regression (stats::lm with poly)
* Logistic regression (stats::glm)
* Ridge, LASSO, elastic net (glmnet)
* Decision trees (rpart)
* Random forests (randomForest)
* Gradient boosting (gbm)
* XGBoost (xgboost)
* Support vector machines (e1071)
* Neural networks (nnet)
* Deep learning (keras, optional)

#### Unsupervised Learning Methods

* Principal Component Analysis (stats::prcomp)
* Multidimensional Scaling (stats, MASS, smacof)
* K-means clustering (stats::kmeans)
* PAM clustering (cluster::pam)
* CLARA clustering (cluster::clara)
* Hierarchical clustering (stats::hclust)
* DBSCAN (dbscan)

#### Additional Features

* `tl_split()` - Train/test splitting with stratification support
* `tl_prepare_data()` - Data preprocessing (scaling, imputation, encoding)
* `tl_evaluate()` - Model evaluation with multiple metrics
* `tl_auto_ml()` - Automated machine learning
* `tl_tune()` - Hyperparameter tuning with grid and random search
* Unified ggplot2-based visualization functions
* Integration workflows combining supervised and unsupervised learning

### Wrapped Packages

tidylearn wraps established R packages including: stats, glmnet, randomForest,
xgboost, gbm, e1071, nnet, rpart, cluster, dbscan, MASS, and smacof.
