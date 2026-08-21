# Plot feature importance for an XGBoost model

Plot feature importance for an XGBoost model

## Usage

``` r
tl_plot_xgboost_importance(model, top_n = 10, importance_type = "gain", ...)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- top_n:

  Number of top features to display (default: 10)

- importance_type:

  Type of importance: "gain", "cover", "frequency"

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost")
  tl_plot_xgboost_importance(model)
}

# }
```
