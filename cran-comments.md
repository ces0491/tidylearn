# tidylearn CRAN Submission

## Minor release (0.3.0)

This release adds a new `tl_read()` family of data ingestion functions,
enabling users to read data from files, databases, and cloud sources
into tidy tibbles — completing the tidylearn workflow from data
extraction to model publishing.

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

### Code quality

* Package-wide lint cleanup (zero lintr issues)
* Replaced unsafe `1:n` patterns with `seq_len()` / `seq_along()`
* Removed unused variables, renamed non-snake_case variables

### New suggested dependencies

readr, readxl, nanoparquet, jsonlite, DBI, RSQLite, RPostgres,
RMariaDB, bigrquery, paws.storage — all on CRAN

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Windows 11 x64, R 4.5.2

## Downstream dependencies

No reverse dependencies.
