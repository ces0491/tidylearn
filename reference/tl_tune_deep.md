# Tune a deep learning model

Tune a deep learning model

## Usage

``` r
tl_tune_deep(
  data,
  formula,
  is_classification = FALSE,
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

  Logical indicating if this is a classification problem

- hidden_layers_options:

  List of vectors defining hidden layer configurations to try

- learning_rates:

  Learning rates to try (default: c(0.01, 0.001, 0.0001))

- batch_sizes:

  Batch sizes to try (default: c(16, 32, 64))

- epochs:

  Number of training epochs (default: 30)

- validation_split:

  Proportion of data for validation (default: 0.2)

- ...:

  Additional arguments

## Value

A list with elements `model` (the best fitted deep learning model),
`best_hidden_layers` (optimal layer configuration),
`best_learning_rate`, `best_batch_size`, and `tuning_results` (a data
frame of all hyperparameter combinations and their validation losses).

## Examples

``` r
if (FALSE) { # \dontrun{
if (requireNamespace("keras", quietly = TRUE)) {
  result <- tl_tune_deep(iris, Species ~ .,
    is_classification = TRUE,
    hidden_layers_options = list(c(10), c(10, 5)),
    learning_rates = c(0.01, 0.001), batch_sizes = c(32),
    epochs = 5)
}
} # }
```
