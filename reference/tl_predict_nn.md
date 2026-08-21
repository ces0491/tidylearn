# Predict using a neural network model

Predict using a neural network model

## Usage

``` r
tl_predict_nn(model, new_data, type = "response", ...)
```

## Arguments

- model:

  A tidylearn neural network model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" (for classification),
  "class" (for classification)

- ...:

  Additional arguments

## Value

Predictions
