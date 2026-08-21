# Plot SHAP summary for XGBoost model

Plot SHAP summary for XGBoost model

## Usage

``` r
tl_plot_xgboost_shap_summary(model, data = NULL, top_n = 10, n_samples = 100)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- data:

  Data for SHAP value calculation (default: NULL, uses training data)

- top_n:

  Number of top features to display (default: 10)

- n_samples:

  Number of samples to use (default: 100, NULL for all)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost")
  tl_plot_xgboost_shap_summary(model, n_samples = 20)
}

# }
```
