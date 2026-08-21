# Fit an XGBoost model

Fit an XGBoost model

## Usage

``` r
tl_fit_xgboost(
  data,
  formula,
  is_classification = FALSE,
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  subsample = 1,
  colsample_bytree = 1,
  min_child_weight = 1,
  gamma = 0,
  alpha = 0,
  lambda = 1,
  early_stopping_rounds = NULL,
  nthread = NULL,
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

- nrounds:

  Number of boosting rounds (default: 100)

- max_depth:

  Maximum depth of trees (default: 6)

- eta:

  Learning rate (default: 0.3)

- subsample:

  Subsample ratio of observations (default: 1)

- colsample_bytree:

  Subsample ratio of columns (default: 1)

- min_child_weight:

  Minimum sum of instance weight needed in a child (default: 1)

- gamma:

  Minimum loss reduction to make a further partition (default: 0)

- alpha:

  L1 regularization term (default: 0)

- lambda:

  L2 regularization term (default: 1)

- early_stopping_rounds:

  Early stopping rounds (default: NULL)

- nthread:

  Number of threads (default: max available)

- verbose:

  Verbose output (default: 0)

- ...:

  Additional arguments to pass to xgb.train()

- compute:

  Compute tier. Either `"cpu"` (default) or `"gpu"`; when `"gpu"`, the
  function passes `device = "cuda"` to `xgb.train()`. Requires an
  xgboost build with CUDA support.

## Value

A fitted XGBoost model
