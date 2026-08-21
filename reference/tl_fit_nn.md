# Fit a neural network model

Fit a neural network model

## Usage

``` r
tl_fit_nn(
  data,
  formula,
  is_classification = FALSE,
  size = 5,
  decay = 0,
  maxit = 100,
  trace = FALSE,
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

- size:

  Number of units in the hidden layer (default: 5)

- decay:

  Weight decay parameter (default: 0)

- maxit:

  Maximum number of iterations (default: 100)

- trace:

  Logical; whether to print progress (default: FALSE)

- ...:

  Additional arguments to pass to nnet()

## Value

A fitted neural network model
