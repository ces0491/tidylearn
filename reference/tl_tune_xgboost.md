# Tune XGBoost hyperparameters

Tune XGBoost hyperparameters

## Usage

``` r
tl_tune_xgboost(
  data,
  formula,
  is_classification = NULL,
  param_grid = NULL,
  cv_folds = 5,
  nrounds = 1000,
  early_stopping_rounds = 10,
  verbose = TRUE,
  ...
)
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the model

- is_classification:

  Logical indicating if this is a classification problem. `NULL`
  (default) reads it from the response, as
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  does: a factor or character response is classification. `FALSE` with
  such a response is an error.

- param_grid:

  Named list of parameter values to try. NULL (default) tries every
  combination of `tl_default_param_grid("xgboost", size = "large")`
  without its `nrounds`, which early stopping chooses here.

- cv_folds:

  Number of cross-validation folds (default: 5)

- nrounds:

  Upper bound on boosting rounds per parameter set (default: 1000).
  Early stopping normally halts well short of it, so this is a ceiling
  rather than a target; lower it to cap the search.

- early_stopping_rounds:

  Early stopping rounds (default: 10)

- verbose:

  Logical indicating whether to print progress (default: TRUE)

- ...:

  Arguments
  [`xgboost::xgb.cv()`](https://rdrr.io/pkg/xgboost/man/xgb.cv.html)
  takes, such as `showsd` or `stratified`, which go to it alone; case
  `weights`, one per row of `data`, which the folds split with the rows;
  and booster parameters held fixed across the grid, such as `nthread`
  or `tree_method`, which join each parameter set and the final fit. A
  value in `param_grid` replaces one given here. The other per-row
  arguments – `subset`, `offset`, `foldid`, `strata` – are refused.

## Value

A `tidylearn_model` object (the refit on full data using the best
hyperparameters, built by
[`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md) so
that it records them in `$spec$args`) with an attribute
`"tuning_results"` containing a list with elements `param_grid`,
`results` (per-combination CV output), `best_params`, `best_iteration`,
`best_score`, and `minimize`.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  # The default grid is every combination of the large xgboost grid
  # without nrounds -- this many:
  default_grid <- tl_default_param_grid("xgboost", size = "large")
  prod(lengths(default_grid[names(default_grid) != "nrounds"]))

  # Name a smaller one to see it run, and cap nrounds so early stopping
  # has less ground to cover. nthread = 2 keeps xgboost from taking
  # every core it is offered.
  tuned <- tl_tune_xgboost(iris, Species ~ .,
    param_grid = list(max_depth = c(2, 4)),
    cv_folds = 3, nrounds = 20, verbose = FALSE, nthread = 2)

  results <- attr(tuned, "tuning_results")
  results$best_params
  results$best_iteration

  # tuned is an ordinary model, refit on all rows at those settings
  predict(tuned, iris[1:5, ])
}
#> # A tibble: 5 × 1
#>   .pred 
#>   <fct> 
#> 1 setosa
#> 2 setosa
#> 3 setosa
#> 4 setosa
#> 5 setosa
# }
```
