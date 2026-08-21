# Predict using a decision tree model

Predict using a decision tree model

## Usage

``` r
tl_predict_tree(model, new_data, type = "response", ...)
```

## Arguments

- model:

  A tidylearn tree model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" or "class" (for
  classification)

- ...:

  Additional arguments

## Value

Predictions
