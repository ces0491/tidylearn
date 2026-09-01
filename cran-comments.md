# tidylearn 0.5.0

This is a minor release of a package already on CRAN (0.4.0). It adds the
security and serialisation groundwork for cloud compute, and fixes a large
group of defects found in a systematic bug hunt across the package — most of
them cases where a degenerate input produced a hang, a silently wrong number,
or an error naming a variable the caller never wrote.

## R CMD check results

0 errors | 0 warnings | 0 notes

A local Windows check intermittently reports one additional NOTE, "checking
for future file timestamps ... unable to verify current time". It appears
only when the check machine cannot reach the time-verification service, and
is unrelated to the package.

## Changes in this version

### New features

* `tl_cloud_consent()` grants or revokes permission, for the rest of the R
  session, to upload training data to the user's Modal account. Without it,
  a cloud fit requires `confirm_upload = TRUE` on every call. The lock is
  never written to disk and does not survive an R restart, and the package
  never prompts interactively, so scripts and CI behave the same way as an
  interactive session.
* Cloud endpoints are read from the `TIDYLEARN_MODAL_ENDPOINT` environment
  variable and validated before any request is built: the scheme must be
  `https` and the host must be on an allowlist that defaults to Modal's own
  domains. `tl_cloud_allow_host()` and `tl_cloud_allowed_hosts()` let Modal
  customers on a custom domain extend it, per session.
* Internal helpers serialise a fitted model to bytes and back for transport
  from a remote worker, with `method = "deep"` handled separately because a
  keras model is a reference to a Python object.

Submission itself is still not wired up. `compute = "cloud"` continues to
error with an explanatory message, and the package makes no network requests
anywhere.

### Bug fixes

The bulk of this release. Grouped by where they surfaced:

* **Degenerate inputs.** A classification forest on a frame with no varying
  predictor did not terminate — randomForest's C loop draws `mtry`
  candidates looking for a split that cannot exist, so the session had to be
  killed. It is now refused before the call. Alongside it: intercept-only
  formulas, single-class responses, a formula response that is not a column
  of the data, unvalidated `cv_folds` and `train_prop`, and unrecognised
  metric names — each of which previously reached a backend and produced a
  message naming neither the argument nor the cause.
* **`method = "forest"` and `method = "svm"` computed their own defaults**
  from the number of columns in the data frame rather than the number of
  predictors in the formula. `mpg ~ wt + hp` on `mtcars` asked randomForest
  for `mtry = 3` of 2 predictors, and asked e1071 for a kernel width of 1/10
  instead of 1/2 with nothing reported. Both now leave the argument unset so
  the wrapped package applies the default it documents.
* **Data leakage.** Preprocessing statistics in cross-validation were
  learned from the full frame rather than the training rows of each fold.
* **Metrics and evaluation**, including the event level: the positive class
  is the second factor level throughout, where yardstick defaults to the
  first.
* **Diagnostics** were computed against data rows the fit had not used.

### Documentation

Corrected five factual errors across the vignettes and README, and rewrote
two articles that had drifted from the register of the other nine.
`model$fit` was documented as the wrapped object throughout; that holds for
supervised methods, but an unsupervised one returns tidied components as
well, so its wrapped object is at `model$fit$model`.

## Test environments

* local: Windows 11 x64, R 4.5.2
* win-builder: R-release (4.6.1, 2026-06-24 ucrt) -- Status: OK
* win-builder: R-devel (2026-08-31 r90457 ucrt) -- Status: OK
* GitHub Actions: ubuntu-latest (R-release and R-devel), macos-latest and
  windows-latest (R-release) -- all OK

## Notes for the reviewer

Three examples use `\dontrun{}`: `tl_plot_deep_architecture()`,
`tl_plot_deep_history()` and `tl_tune_deep()`. All three fit keras models,
which need a working Python and TensorFlow installation that a check machine
will not have. They are already guarded with `requireNamespace("keras")`;
`\dontrun{}` rather than `\donttest{}` is used because the failure mode is a
Python-level error rather than a slow run. Every other example that needs a
Suggests package is wrapped in `\donttest{}` with a `requireNamespace()`
guard.

## Downstream dependencies

There are no reverse dependencies.
