# tidylearn CRAN Submission

## Minor release (0.2.0)

This release adds a new family of `tl_table_*()` functions for producing
formatted `gt` tables from tidylearn models — mirroring the existing
`plot()` interface but for report-ready tables. Changes:

* Added `tl_table()` dispatcher and 8 exported table functions:
  `tl_table_metrics()`, `tl_table_coefficients()`, `tl_table_confusion()`,
  `tl_table_importance()`, `tl_table_variance()`, `tl_table_loadings()`,
  `tl_table_clusters()`, `tl_table_comparison()`
* `gt` added as a suggested dependency (not required for core functionality)
* New "Reporting with tidylearn" vignette

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Windows 11 x64, R 4.5.2

## Downstream dependencies

No reverse dependencies.
