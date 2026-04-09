# tidylearn CRAN Submission

## Minor release (0.3.0)

This release adds a new `tl_read()` family of data ingestion functions,
fixes 12 bugs across workflows, pipelines, interactions, and visualizations,
and substantially expands test coverage.

### New features

* `tl_read()` dispatcher with auto-format detection from file extensions,
  URL patterns, and connection strings
* File format readers: `tl_read_csv()`, `tl_read_tsv()`, `tl_read_excel()`,
  `tl_read_parquet()`, `tl_read_json()`, `tl_read_rds()`, `tl_read_rdata()`
* Database readers: `tl_read_db()`, `tl_read_sqlite()`, `tl_read_postgres()`,
  `tl_read_mysql()`, `tl_read_bigquery()`
* Cloud/API readers: `tl_read_s3()`, `tl_read_github()`, `tl_read_kaggle()`
* Multi-file reading: `tl_read_dir()`, `tl_read_zip()`, and vector-of-paths
  support in `tl_read()`
* All readers return `tidylearn_data` objects (tibble subclass with metadata)
* All backend packages are suggested dependencies only
* New "Data Ingestion with tidylearn" vignette

### Bug fixes

* Fixed `tl_transfer_learning()` hanging due to `.obs_id` column from PCA
  output being included in the supervised formula
* Fixed `tl_run_pipeline()` failing when all CV metrics were NA (`scale()`
  returning matrix columns, NaN metric handling)
* Overhauled `tl_auto_ml()` time budget enforcement -- budget now gates
  which models are attempted and whether CV runs
* Fixed `tl_interaction_effects()` and `tl_plot_interaction()` predict
  interface mismatches with tidylearn's `.pred` tibble output
* Fixed `tl_plot_intervals()` calling non-existent function; now uses
  `stats::predict()` directly for confidence/prediction intervals
* Fixed `tl_plot_svm_boundary()` failing on `response ~ .` formulas and
  on discrete class predictions
* Fixed `tl_plot_svm_tuning()` passing NULL entries to `e1071::tune()`
* Fixed `tl_plot_xgboost_shap_summary()` row count mismatch when sampling
* Fixed classification auto-detection silently misclassifying numeric
  responses with few unique values
* Fixed `tl_check_assumptions()` crashing when optional test packages
  were not installed
* Fixed SVM default gamma to use predictor count only

### Tests

* Added 150 new tests across visualization, tuning, and diagnostics modules
* Total test count: 509 passing, 0 failures

### Code quality

* Package-wide lint cleanup (zero lintr issues)
* Replaced unsafe `1:n` patterns with `seq_len()` / `seq_along()`

### New suggested dependencies

readr, readxl, nanoparquet, jsonlite, DBI, RSQLite, RPostgres,
RMariaDB, bigrquery, paws.storage -- all on CRAN

## R CMD check results

0 errors | 0 warnings | 1 note

The only NOTE is "unable to verify current time" — a transient network
issue unrelated to the package.

## Test environments

* local: Windows 11 x64, R 4.5.2
* GitHub Actions: ubuntu-latest (R release, R devel),
  macos-latest (R release), windows-latest (R release)

## Downstream dependencies

No reverse dependencies.
