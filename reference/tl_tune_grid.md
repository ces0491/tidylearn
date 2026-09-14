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
`param_grid`, `results`, `best_params`, `best_metric`, `metric`, and
`maximize`.

`results` has one row per evaluated combination: `mean_metric` (the mean
over the folds that produced a score), `n_folds_ok` (how many of the
`folds` did), and a column per parameter. A parameter with a
vector-valued candidate, such as `hidden_layers`, is a list column.

Only combinations with `n_folds_ok` equal to `folds` are eligible to be
best, since a mean over the folds that happened to succeed is not
comparable with a mean over all of them. If no combination completed
every fold, the best of those scored on the most folds is used, with a
warning. If every combination failed in every fold, the function stops.

For `method = "forest"`, an `mtry` above the number of predictors is
capped at that number, with a warning, and duplicate combinations that
result are evaluated once.

## Examples

``` r
# \donttest{
model <- tl_tune_grid(iris, Species ~ ., method = "tree",
  param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  folds = 2, verbose = FALSE)
# }
```
