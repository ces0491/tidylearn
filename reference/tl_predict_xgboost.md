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

  Boosting iterations to predict with, as `c(start, end)` – base-1 and
  inclusive of both ends, so `c(1, 20)` predicts from the first twenty
  iterations and `end` may not exceed the number fitted. NULL (default)
  uses every iteration.

- ntreelimit:

  Deprecated. Use `iterationrange` instead. `ntreelimit = n` is
  translated to `c(1, n)`.

- ...:

  Additional arguments

## Value

Predictions
