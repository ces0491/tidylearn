# Compare models from a pipeline

Compare models from a pipeline

## Usage

``` r
tl_compare_pipeline_models(pipeline, metrics = NULL)
```

## Arguments

- pipeline:

  A tidylearn pipeline object with results

- metrics:

  Character vector of metrics to compare (if NULL, uses all available)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) object
showing a faceted bar chart comparing metric values across models, with
the best model highlighted.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .,
  models = list(
    tree = list(method = "tree"),
    forest = list(method = "forest", ntree = 100)
  ),
  evaluation = list(validation = "cv", cv_folds = 3))
pipe <- tl_run_pipeline(pipe, verbose = FALSE)

tl_compare_pipeline_models(pipe)


# Restrict the comparison to one metric
tl_compare_pipeline_models(pipe, metrics = "accuracy")

# }
```
