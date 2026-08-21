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
metric specified in `evaluation$best_metric`.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .,
  models = list(tree = list(method = "tree")),
  evaluation = list(metrics = "accuracy", validation = "cv",
    cv_folds = 2, best_metric = "accuracy"))
pipe <- tl_run_pipeline(pipe, verbose = FALSE)
best <- tl_get_best_model(pipe)
# }
```
