# Make predictions using a pipeline

Make predictions using a pipeline

## Usage

``` r
tl_predict_pipeline(
  pipeline,
  new_data,
  type = "response",
  model_name = NULL,
  ...
)
```

## Arguments

- pipeline:

  A tidylearn pipeline object with results

- new_data:

  A data frame containing the new data

- type:

  Type of prediction (default: "response")

- model_name:

  Name of model to use (if NULL, uses the best model)

- ...:

  Additional arguments passed to predict

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with a
`.pred` column containing predictions from the selected (or best)
pipeline model, after applying the same preprocessing steps used during
training.
