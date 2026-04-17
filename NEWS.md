# tidylearn 0.3.0.9000

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

## Tests

* Added 10 tests for `tidy_gower()` / `tidy_dist(..., method = "gower")`
  covering: return type and metadata, symmetry and self-distance, identical
  rows, hand-verified numeric / categorical / ordered / mixed-type distances,
  NA skipping, custom weights and constant-column denominator behaviour.


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
