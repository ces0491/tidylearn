# Predict using a support vector machine model

Predict using a support vector machine model

## Usage

``` r
tl_predict_svm(model, new_data, type = "response", ...)
```

## Arguments

- model:

  A tidylearn SVM model object

- new_data:

  A data frame containing the new data

- type:

  Type of prediction: "response" (default), "prob" (for classification)

- ...:

  Additional arguments

## Value

Predictions
