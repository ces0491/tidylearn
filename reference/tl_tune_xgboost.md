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

- early_stopping_rounds:

  Early stopping rounds (default: 10)

- verbose:

  Logical indicating whether to print progress (default: TRUE)

- ...:

  Additional arguments

## Value

A `tidylearn_model` object (the refit on full data using the best
hyperparameters) with an attribute `"tuning_results"` containing a list
with elements `param_grid`, `results` (per-combination CV output),
`best_params`, `best_iteration`, `best_score`, and `minimize`.
