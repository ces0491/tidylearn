# Tune a neural network model

Tune a neural network model

## Usage

``` r
tl_tune_nn(
  data,
  formula,
  is_classification = NULL,
  sizes = c(1, 2, 5, 10),
  decays = c(0, 0.001, 0.01, 0.1),
  folds = 5,
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

- sizes:

  Vector of hidden layer sizes to try

- decays:

  Vector of weight decay parameters to try

- folds:

  Number of cross-validation folds (default: 5)

- ...:

  Additional arguments to pass to nnet(). `maxit` (default 100) and
  `trace` (default `FALSE`) replace the values used otherwise. Arguments
  with one value per row – `weights`, `subset`, `offset`, `foldid`,
  `strata` – are refused, since each fold fits a subset of the rows.

## Value

A list with elements `model` (the best fitted `nnet` model), `best_size`
(optimal hidden-layer size), `best_decay` (optimal weight decay), and
`tuning_results` (a data frame of all parameter combinations and their
cross-validated errors: the misclassification rate for classification,
the mean squared error for regression).

## Examples

``` r
# \donttest{
tuned <- tl_tune_nn(iris, Species ~ .,
  is_classification = TRUE,
  sizes = c(2, 5), decays = c(0, 0.01), folds = 3)

tuned$best_size
#> [1] 2
tuned$best_decay
#> [1] 0
tuned$tuning_results
#>   size decay      error
#> 1    2  0.00 0.03333333
#> 2    5  0.00 0.03333333
#> 3    2  0.01 0.04000000
#> 4    5  0.01 0.03333333

# The grid this searched, drawn as a heatmap
tl_plot_nn_tuning(tuned)

# }
```
