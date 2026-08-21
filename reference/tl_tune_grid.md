# Tune hyperparameters for a model using grid search

Tune hyperparameters for a model using grid search

## Usage

``` r
tl_tune_grid(
  data,
  formula,
  method,
  param_grid,
  folds = 5,
  metric = NULL,
  maximize = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the model

- method:

  The modeling method to tune

- param_grid:

  A named list of parameter values to tune

- folds:

  Number of cross-validation folds

- metric:

  Metric to optimize

- maximize:

  Logical; whether to maximize (TRUE) or minimize (FALSE) the metric

- verbose:

  Logical; whether to print progress

- ...:

  Additional arguments passed to tl_model

## Value

A tidylearn model object fitted with the best hyperparameters. Tuning
results are stored as an attribute `"tuning_results"`, a list containing
`param_grid`, `results` (data frame of all evaluated combinations),
`best_params`, `best_metric`, `metric`, and `maximize`.

## Examples

``` r
# \donttest{
model <- tl_tune_grid(iris, Species ~ ., method = "tree",
  param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  folds = 2, verbose = FALSE)
# }
```
