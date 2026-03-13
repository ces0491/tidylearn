# tidylearn CRAN Submission

## Patch release (0.1.1)

This is a patch release fixing a bug reported in
[#1](https://github.com/ces0491/tidylearn/issues/1). Changes:

* Added missing `tl_plot_model()` and `tl_plot_unsupervised()` internal
  dispatcher functions so that `plot()` works on tidylearn model objects
* Fixed column name mismatch (`$prediction` -> `$.pred`) in
  `tl_plot_actual_predicted()`, `tl_plot_residuals()`,
  `tl_plot_confusion()`, and `tl_dashboard()`

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* Windows 11 x64, R 4.4.3
* GitHub Actions: ubuntu-latest (R-devel, R-release), windows-latest (R-release), macos-latest (R-release)

## Downstream dependencies

No reverse dependencies.
