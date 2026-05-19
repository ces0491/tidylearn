# tidylearn 0.3.1

This is a patch release of a package already on CRAN (0.3.0). It fixes three
pre-existing bugs, optimises the Gower distance computation, and removes seven
unused packages from `Suggests`.

## R CMD check results

0 errors | 0 warnings | 0 notes on win-builder (R-devel).

A local Windows check additionally reports one NOTE, "unable to verify
current time". This is environmental: the local check machine could not
reach the time-verification web service. It is unrelated to the package and
does not appear on machines with network access to that service.

## Changes in this version

### Bug fixes

* `tl_reduce_dimensions()` no longer returns the internal `.obs_id` row
  identifier in its `$data` element. Previously that high-cardinality
  identifier could be carried into a downstream `response ~ .` model as a
  predictor, which made tree-based fits effectively non-terminating.
* `print()` and `summary()` no longer error on the model objects returned
  by `tl_step_selection()` and `tl_tune_xgboost()`. Both objects are now
  constructed consistently with `tl_model()` (with a `spec$paradigm` field
  and the `tidylearn_supervised` class).
* `tidy_gower()` no longer errors on single-row input. The pairwise loop
  used `1:(n - 1)`, which is the invalid sequence `1:0` when `n` is 1; it
  now uses `seq_len()` and returns an empty `dist` object.

### Other changes

* Optimised `tidy_gower()`: pairwise Gower distance is reduced from
  O(n^2 * p) to O(n^2 + p) work, with no change in results.
* Removed seven packages from `Suggests` that were not referenced anywhere
  in the package (caret, mclust, onnx, parsnip, recipes, reticulate,
  workflows).

## Test environments

* local: Windows 11 x64, R 4.5.2
* win-builder: R-devel (R Under development, r90061)

## Downstream dependencies

There are no reverse dependencies.
