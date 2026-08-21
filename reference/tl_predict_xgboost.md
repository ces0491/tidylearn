# Predict using an XGBoost model

Predict using an XGBoost model

## Usage

``` r
tl_predict_xgboost(
  model,
  new_data,
  type = "response",
  iterationrange = NULL,
  ntreelimit = NULL,
  ...
)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" (for classification),
  "class" (for classification)

- iterationrange:

  Boosting iterations to predict with, as `c(start, end)` (default:
  NULL, uses every iteration).

- ntreelimit:

  Deprecated. Use `iterationrange` instead.

- ...:

  Additional arguments

## Value

Predictions
