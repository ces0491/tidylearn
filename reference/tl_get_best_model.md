# Get the best model from a pipeline

Get the best model from a pipeline

## Usage

``` r
tl_get_best_model(pipeline)
```

## Arguments

- pipeline:

  A tidylearn pipeline object with results

## Value

The best `tidylearn_model` object from the pipeline, selected by the
metric specified in `evaluation$best_metric`. The model was fitted on
the preprocessed training data, so under the default preprocessing it
expects predictors imputed and standardised the same way. Predict
through
[`tl_predict_pipeline`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md),
which replays that preprocessing on raw rows;
[`predict()`](https://rdrr.io/r/stats/predict.html) on the model itself
reads raw values as if they were already standardised, and returns wrong
predictions without a warning.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .,
  models = list(tree = list(method = "tree")),
  evaluation = list(metrics = "accuracy", validation = "cv",
    cv_folds = 2, best_metric = "accuracy"))
pipe <- tl_run_pipeline(pipe, verbose = FALSE)
best <- tl_get_best_model(pipe)
best$spec$method
#> [1] "tree"

# Predict through the pipeline, which preprocesses the new rows first
tl_predict_pipeline(pipe, iris[c(1, 51, 101), ])
#> # A tibble: 3 × 1
#>   .pred     
#>   <fct>     
#> 1 setosa    
#> 2 versicolor
#> 3 virginica 
# }
```
