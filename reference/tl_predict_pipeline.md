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

## Examples

``` r
# \donttest{
train <- iris[c(1:40, 51:90, 101:140), ]
test <- iris[c(41:50, 91:100, 141:150), ]

pipe <- tl_pipeline(train, Species ~ .,
  models = list(
    tree = list(method = "tree"),
    forest = list(method = "forest", ntree = 100)
  ),
  evaluation = list(validation = "cv", cv_folds = 3))
pipe <- tl_run_pipeline(pipe, verbose = FALSE)

# The best model, with the preprocessing learned on the training rows
tl_predict_pipeline(pipe, test)
#> # A tibble: 30 × 1
#>    .pred 
#>    <fct> 
#>  1 setosa
#>  2 setosa
#>  3 setosa
#>  4 setosa
#>  5 setosa
#>  6 setosa
#>  7 setosa
#>  8 setosa
#>  9 setosa
#> 10 setosa
#> # ℹ 20 more rows

# Or a named candidate instead of the winner
tl_predict_pipeline(pipe, test, model_name = "tree")
#> # A tibble: 30 × 1
#>    .pred 
#>    <fct> 
#>  1 setosa
#>  2 setosa
#>  3 setosa
#>  4 setosa
#>  5 setosa
#>  6 setosa
#>  7 setosa
#>  8 setosa
#>  9 setosa
#> 10 setosa
#> # ℹ 20 more rows
# }
```
