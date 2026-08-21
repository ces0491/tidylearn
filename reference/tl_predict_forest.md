# Predict using a random forest model

Predict using a random forest model

## Usage

``` r
tl_predict_forest(model, new_data, type = "response", ...)
```

## Arguments

- model:

  A tidylearn forest model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" (for classification)

- ...:

  Additional arguments

## Value

Predictions
