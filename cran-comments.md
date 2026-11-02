# tidylearn 0.7.0

## About this release

This is a minor release of a package already on CRAN (0.6.0, published
2026-10-08). It follows 0.6.0 by under four weeks because it corrects results
0.6.0 reports without an error or warning:

* `tl_evaluate()` on an unsupervised model returned one placeholder row,
  `metric = "completed"`, `value = 1`, whatever it was given. It is now an
  error that names the functions which assess a clustering.
* `tl_check_assumptions()` read rounding error as a verdict, so a perfect
  line reported three violated assumptions. A check the fit cannot support
  now reports `NA`.
* `tl_compute_advisor()` divided its runtime estimates by the machine's
  physical cores, although only two of its thirteen methods use more than
  one. On a 16-core laptop that left forest and tree estimates 14x to 26x
  under the measured times.

The `"deep"` method moves from 'keras' to 'keras3', because 'keras' reports
itself as deprecated when loaded. Both are optional, so 'keras3' replaces
'keras' in Suggests. 'later' joins Suggests for one test, and 'parallel'
leaves Imports.

`tl_plot_tuning_results()` no longer calls `summary()` of an `aov()`, which
gained a warning in R 4.6 that this function could trigger.

NEWS.md lists four changes that can stop an existing call under Breaking
Changes, and opens in bold each bug fix that changes a result.

## R CMD check results

0 errors | 0 warnings | 0 notes from the package

The local Windows check gives two NOTEs that come from the machine: "unable
to verify current time", when the time service cannot be reached, and a
`lastMiKTeXException` file that the local MiKTeX installation leaves in the
temp directory.

## Test environments

Each run below used the source of the 0.7.0 tarball submitted here.

* Local: Windows 11 x64, R 4.5.2 (2025-10-31 ucrt), `--as-cran`,
  2026-11-02: 0 errors, 0 warnings, the 2 NOTEs above.
* Docker: rocker/r-ver:4.6.2 on Ubuntu 26.04, 16 cores, OpenBLAS (pthread),
  current CRAN packages, `--as-cran` with the example, test and vignette
  CPU-to-elapsed thresholds at 2.5, 2026-11-02: Status OK. The example, test
  and vignette steps ran at a CPU-to-elapsed ratio of 1.5 or less.
* Docker: the same image with only Imports, 'testthat' and 'withr' installed,
  vignettes skipped, 2026-11-02: Status OK.
* win-builder: R-release (4.6.2, 2026-10-29 ucrt), 2026-11-02: Status OK.
* win-builder: R-devel (2026-11-01 r90825 ucrt), 2026-11-02: Status OK.
* GitHub Actions, `--as-cran`, 2026-11-02: ubuntu-latest (R-release and
  R-devel), macos-latest and windows-latest (R-release): Status OK on each.
* Docker: rocker/r-ver:4.1.3, the minimum R version, with packages from a
  2022-12-01 snapshot (ggplot2 3.4.0, rlang 1.0.6), 2026-11-02: installs, and
  a script exercising fitting, prediction, cross-validation, plotting and
  the changed diagnostics, advisor and evaluation functions runs. This was
  an installation test, not a full R CMD check.

## Notes for the reviewer

Nine examples use `\dontrun{}`.

* `tl_plot_deep_architecture()`, `tl_plot_deep_history()` and
  `tl_tune_deep()` fit 'keras3' models, which need a working Python and
  TensorFlow installation that a check machine will not have.
* `tl_read_bigquery()`, `tl_read_github()`, `tl_read_kaggle()`,
  `tl_read_mysql()`, `tl_read_postgres()` and `tl_read_s3()` read from a
  remote service. `tl_read_github()` needs network access; the other five
  also need credentials, a running database server or the Kaggle command
  line tool.

## Downstream dependencies

There are no reverse dependencies on CRAN (checked with
`tools::package_dependencies(reverse = TRUE)` on 2026-11-02).
