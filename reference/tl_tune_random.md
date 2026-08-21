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

  A named list of parameter spaces to sample from

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
`param_space`, `results` (data frame of all evaluated iterations),
`best_params`, `best_metric`, `metric`, and `maximize`.

## Examples

``` r
# \donttest{
model <- tl_tune_random(mtcars, mpg ~ ., method = "tree",
  param_space = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  n_iter = 3, folds = 2, verbose = FALSE)
# }
```
