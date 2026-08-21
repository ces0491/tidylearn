# Predict using a gradient boosting model

Predict using a gradient boosting model

## Usage

``` r
tl_predict_boost(model, new_data, type = "response", n.trees = NULL, ...)
```

## Arguments

- model:

  A tidylearn boost model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" (for classification)

- n.trees:

  Number of trees to use for prediction (if NULL, uses optimal number)

- ...:

  Additional arguments

## Value

Predictions
