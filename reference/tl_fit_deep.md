# Fit a deep learning model

Fit a deep learning model

## Usage

``` r
tl_fit_deep(
  data,
  formula,
  is_classification = FALSE,
  hidden_layers = c(32, 16),
  activation = "relu",
  dropout = 0.2,
  epochs = 30,
  batch_size = 32,
  validation_split = 0.2,
  learning_rate = NULL,
  verbose = 0,
  ...,
  compute = "cpu"
)
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the model

- is_classification:

  Logical indicating if this is a classification problem

- hidden_layers:

  Vector of units in each hidden layer (default: c(32, 16))

- activation:

  Activation function for hidden layers (default: "relu")

- dropout:

  Dropout rate for regularization (default: 0.2)

- epochs:

  Number of training epochs (default: 30)

- batch_size:

  Batch size for training (default: 32)

- validation_split:

  Proportion of data for validation

- learning_rate:

  Optimizer learning rate. NULL (default) leaves keras's own adam
  default in place. (default: 0.2)

- verbose:

  Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
  (default: 0)

- ...:

  Additional arguments

- compute:

  Compute tier. Either `"cpu"` (default) or `"gpu"`. GPU usage is
  handled automatically by the underlying tensorflow runtime when CUDA
  is configured; this argument is accepted for API consistency with the
  rest of tidylearn but does not itself change the keras model setup.
  The expectation is that the caller has already resolved the compute
  tier via
  [`tl_compute_advisor`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  / `tl_resolve_compute`.

## Value

A fitted deep learning model
