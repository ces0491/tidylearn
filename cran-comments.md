# tidylearn 0.6.0

## About this release

This is a minor release of a package already on CRAN (0.5.0). It fixes the
defects found in a review of the whole package, and adds
`tl_coefficients()`, which returns a model's coefficients and their
confidence intervals as a tibble. It follows 0.5.0 by five weeks because
a number of the fixes correct values 0.5.0 returned without an error or
warning, such as `"svm"` predictions that were dropped and misaligned
when the new data held a missing value.

Some existing calls now return different values. NEWS.md lists them
under Breaking Changes, and opens in bold each bug fix that changes a
number 0.5.0 returned. The two a user is most likely to notice:

* `predict()` on a `"ridge"`, `"lasso"` or `"elastic_net"` model uses
  `lambda_1se`. It used `lambda_min`, while the coefficients and
  importance the package reports for the same model were at
  `lambda_1se`.
* Importance for those three methods is each coefficient multiplied by
  the standard deviation of its column, so it no longer changes with the
  units a predictor is measured in.

The minimum versions rise to R 4.1.0, because the package, README and
vignettes use the native `|>` pipe, and to ggplot2 3.4.0, for the
`linewidth` aesthetic. The rlang floor rises to 1.0.0, which ggplot2
3.4.0 already requires.

## R CMD check results

0 errors | 0 warnings | 0 notes from the package

The local checks give NOTEs that come from the machines: "unable to
verify current time", when the time service cannot be reached, and on
Windows a `lastMiKTeXException` file that the local MiKTeX installation
leaves in the temp directory.

## Test environments

Each run below checked the 0.6.0 tarball submitted here.

* Local: Windows 11 x64, R 4.5.2 (2025-10-31 ucrt), `--as-cran`,
  2026-10-08: 0 errors, 0 warnings, the 2 NOTEs above.
* Docker: rocker/r-ver:4.5.2 on Ubuntu 24.04, 16 cores, OpenBLAS
  (pthread), `--as-cran` with the example, test and vignette CPU-to-elapsed
  thresholds at 2.5, 2026-10-08: 0 errors, 0 warnings, the time NOTE
  above. This is the configuration that reproduced the CPU-time NOTE
  behind 0.5.0's pre-test rejections; every timing step ran at a
  CPU-to-elapsed ratio of 1.3 or less.
* Docker: rocker/r-ver:4.1.3, the minimum R version, with packages from a
  2022-12-01 snapshot (ggplot2 3.4.0, rlang 1.0.6), 2026-10-08: installs,
  and a script exercising fitting, prediction, splitting,
  cross-validation and plotting runs. This was an installation test, not
  a full R CMD check.

## Notes for the reviewer

Nine examples use `\dontrun{}`.

* `tl_plot_deep_architecture()`, `tl_plot_deep_history()` and
  `tl_tune_deep()` fit keras models, which need a working Python and
  TensorFlow installation that a check machine will not have. These three
  are unchanged from 0.5.0.
* `tl_read_bigquery()`, `tl_read_github()`, `tl_read_kaggle()`,
  `tl_read_mysql()`, `tl_read_postgres()` and `tl_read_s3()` read from a
  remote service. `tl_read_github()` needs network access; the other five
  also need credentials, a running database server or the Kaggle command
  line tool. In 0.5.0 these pages held commented-out code inside
  `\donttest{}`; they now show the calls themselves.

The other readers' examples now run, on temporary files or on files that
readr and readxl ship.

## Downstream dependencies

There are no reverse dependencies on CRAN (checked with
`tools::package_dependencies(reverse = TRUE)` on 2026-10-08).
