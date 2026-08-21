# Tune a neural network model

Tune a neural network model

## Usage

``` r
tl_tune_nn(
  data,
  formula,
  is_classification = FALSE,
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

  Logical indicating if this is a classification problem

- sizes:

  Vector of hidden layer sizes to try

- decays:

  Vector of weight decay parameters to try

- folds:

  Number of cross-validation folds (default: 5)

- ...:

  Additional arguments to pass to nnet()

## Value

A list with elements `model` (the best fitted `nnet` model), `best_size`
(optimal hidden-layer size), `best_decay` (optimal weight decay), and
`tuning_results` (a data frame of all parameter combinations and their
cross-validated errors).
