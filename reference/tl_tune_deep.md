# Tune a deep learning model

Tune a deep learning model

## Usage

``` r
tl_tune_deep(
  data,
  formula,
  is_classification = NULL,
  hidden_layers_options = list(c(32), c(64, 32), c(128, 64, 32)),
  learning_rates = c(0.01, 0.001, 1e-04),
  batch_sizes = c(16, 32, 64),
  epochs = 30,
  validation_split = 0.2,
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

- hidden_layers_options:

  List of vectors defining hidden layer configurations to try

- learning_rates:

  Learning rates to try (default: c(0.01, 0.001, 0.0001))

- batch_sizes:

  Batch sizes to try (default: c(16, 32, 64))

- epochs:

  Number of training epochs (default: 30)

- validation_split:

  Proportion of the rows held out to score each configuration on, drawn
  at random (default: 0.2). Every configuration is scored on the same
  rows.

- ...:

  Additional arguments passed to keras's fit() for every configuration;
  `verbose` (default 0) replaces the value used otherwise. Arguments
  with one value per row – `weights`, `subset`, `offset`, `foldid`,
  `strata` – are refused, since each configuration is fitted on part of
  the rows.

## Value

A list with elements `model` (the best configuration refitted as a
`tidylearn_model`, so
[`predict()`](https://rdrr.io/r/stats/predict.html) and the deep plots
take it; the keras model is at `$model$fit$model`), `best_hidden_layers`
(optimal layer configuration), `best_learning_rate`, `best_batch_size`,
and `tuning_results` (a data frame of all hyperparameter combinations
and their validation losses).

## Examples

``` r
if (FALSE) { # \dontrun{
if (requireNamespace("keras", quietly = TRUE)) {
  result <- tl_tune_deep(iris, Species ~ .,
    hidden_layers_options = list(c(10), c(10, 5)),
    learning_rates = c(0.01, 0.001), batch_sizes = c(32),
    epochs = 5)
  predict(result$model, iris[1:5, ])
}
} # }
```
