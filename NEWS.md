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
