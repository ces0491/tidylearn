# Tune XGBoost hyperparameters

Tune XGBoost hyperparameters

## Usage

``` r
tl_tune_xgboost(
  data,
  formula,
  is_classification = FALSE,
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

  Logical indicating if this is a classification problem

- param_grid:

  Named list of parameter values to try

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

  Additional arguments passed to
  [`xgboost::xgb.cv()`](https://rdrr.io/pkg/xgboost/man/xgb.cv.html)

## Value

A `tidylearn_model` object (the refit on full data using the best
hyperparameters) with an attribute `"tuning_results"` containing a list
with elements `param_grid`, `results` (per-combination CV output),
`best_params`, `best_iteration`, `best_score`, and `minimize`.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  # The default grid is 216 combinations; name a smaller one to see it
  # run, and cap nrounds so early stopping has less ground to cover
  tuned <- tl_tune_xgboost(iris, Species ~ .,
    is_classification = TRUE,
    param_grid = list(max_depth = c(2, 4), eta = c(0.1, 0.3)),
    cv_folds = 3, nrounds = 50, verbose = FALSE)

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
