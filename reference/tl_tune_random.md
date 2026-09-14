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

  The modeling method to tune

- param_space:

  A named list of parameter spaces to sample from. Each element is read
  by its type and length:

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

  Number of random parameter combinations to try

- folds:

  Number of cross-validation folds

- metric:

  Metric to optimize

- maximize:

  Logical; whether to maximize (TRUE) or minimize (FALSE) the metric

- verbose:

  Logical; whether to print progress

- seed:

  Random seed for reproducibility

- ...:

  Additional arguments passed to tl_model

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
