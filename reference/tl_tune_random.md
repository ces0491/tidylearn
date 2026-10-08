# Tune hyperparameters using random search

Tune hyperparameters using random search

## Usage

``` r
tl_tune_random(
  data,
  formula,
  method,
  param_space,
  n_iter = 10,
  folds = 5,
  metric = NULL,
  maximize = NULL,
  verbose = TRUE,
  seed = NULL,
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

- param_space:

  A named list of parameter spaces to sample from, one element for each
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  argument to tune, named after it. Each element is read by its type and
  length:

  a function

  :   called with no arguments to draw one value

  a list

  :   a set of candidates, each drawn whole, e.g.
      `list(c(10), c(20, 10))` for `hidden_layers`

  `c(min, max, "log")`

  :   log-uniform draw between `min` and `max`

  a single value

  :   used as given in every iteration

  two whole numbers

  :   integer range, e.g. `c(10, 20)` draws from 10:20

  three or more numbers

  :   a discrete set, sampled from as given, whether or not they are
      whole

  two other numbers

  :   uniform draw between them, e.g. `c(0.01, 0.1)`

  character or factor

  :   categorical, sampled from as given

  logical

  :   sampled from the values given

- n_iter:

  Number of random parameter combinations to try, a whole number of at
  least 1

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

  Metric to optimize, as for
  [`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)

- maximize:

  Logical; whether to maximize (TRUE) or minimize (FALSE) the metric.
  `NULL`, the default, follows the metric, as for
  [`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md).

- verbose:

  Logical; whether to print progress

- seed:

  Random seed for reproducibility

- ...:

  Additional arguments passed to
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  for every fold and for the final fit. Per-row arguments are refused,
  as for
  [`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md).

## Value

A tidylearn model object fitted with the best hyperparameters. Tuning
results are stored as an attribute `"tuning_results"`, a list containing
`param_space`, `results`, `best_params`, `best_metric`, `metric`, and
`maximize`.

`results` has one row per iteration: `iteration`, `mean_metric`,
`n_folds_ok`, and a column per parameter, as described for
[`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md).
The best parameters are chosen by the same rules, and `mtry` is capped
the same way; duplicate draws are kept, so there are always `n_iter`
rows.

## Examples

``` r
# \donttest{
model <- tl_tune_random(mtcars, mpg ~ ., method = "tree",
  param_space = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  n_iter = 3, folds = 2, verbose = FALSE)
# }
```
