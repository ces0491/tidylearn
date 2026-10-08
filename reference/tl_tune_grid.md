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

  The modeling method to tune, one of the supervised methods
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  fits

- param_grid:

  A named list of candidate values, one element for each
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  argument to tune, named after it

- folds:

  Number of cross-validation folds, a whole number between 2 and
  `nrow(data)`. `nrow(data)` leaves each row out in turn, and each fold
  then scores a single prediction. `"accuracy"`, `"mae"`, `"mse"` and
  `"mape"` average to their values over the left-out predictions. The
  average `"rmse"` is the mean absolute error; `"precision"`,
  `"recall"`, `"sensitivity"`, `"specificity"` and `"f1"` are undefined
  on the folds whose one row gives them nothing to divide by; and
  `"rsq"`, `"auc"` and `"pr_auc"` are undefined on every fold. A run
  scoring any of these warns once.

- metric:

  Metric to optimize: one of the names
  [`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  computes for the task. Defaults to `"accuracy"` for classification and
  `"rmse"` for regression.

- maximize:

  Logical; whether to maximize (TRUE) or minimize (FALSE) the metric.
  `NULL`, the default, maximizes every metric but the error metrics
  `"rmse"`, `"mse"`, `"mae"` and `"mape"`.

- verbose:

  Logical; whether to print progress

- ...:

  Additional arguments passed to
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  for every fold and for the final fit. Arguments holding one value per
  row of `data` – `weights`, `subset`, `offset`, `foldid` and `strata` –
  are refused, since they cannot follow the rows into a fold.

## Value

A tidylearn model object fitted with the best hyperparameters. Tuning
results are stored as an attribute `"tuning_results"`, a list containing
`param_grid`, `results`, `best_params`, `best_metric`, `metric`, and
`maximize`.

`results` has one row per evaluated combination: `mean_metric` (the mean
over the folds that produced a score), `n_folds_ok` (how many of the
`folds` did), and a column per parameter. A parameter with a
vector-valued candidate, such as `hidden_layers`, is a list column. A
fold produces no score when its fit fails, when the metric is undefined
on it – `"auc"` on a fold holding one class, or `"precision"` on one
where nothing is predicted positive – or when none of its rows can be
scored, as when every predictor is missing there, which is warned about.

Only combinations with `n_folds_ok` equal to `folds` are eligible to be
best, since a mean over the folds that happened to succeed is not
comparable with a mean over all of them. If no combination was scored on
every fold, the best of those scored on the most folds is used, with a
warning saying whether fits failed or the metric was undefined. If no
combination was scored on any fold, the function stops.

For `method = "forest"`, an `mtry` above the number of predictors is
capped at that number, with a warning, and duplicate combinations that
result are evaluated once. Predictors are counted as the forest is
fitted on them, so a column removed with `- id` is not one, and a
matrix-valued term such as `poly(hp, 2)` is one per column.

## Examples

``` r
# \donttest{
model <- tl_tune_grid(iris, Species ~ ., method = "tree",
  param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  folds = 2, verbose = FALSE)
# }
```
