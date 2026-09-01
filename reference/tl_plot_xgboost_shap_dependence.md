# Plot SHAP dependence for a specific feature

Plot SHAP dependence for a specific feature

## Usage

``` r
tl_plot_xgboost_shap_dependence(
  model,
  feature,
  interaction_feature = NULL,
  data = NULL,
  n_samples = 100
)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- feature:

  Feature name to plot

- interaction_feature:

  Feature to use for coloring (default: NULL)

- data:

  Data for SHAP value calculation (default: NULL, uses training data)

- n_samples:

  Number of samples to use (default: 100, NULL for all)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 10)

  tl_plot_xgboost_shap_dependence(model, feature = "Petal.Length")

  # Colour the points by a second feature to read the interaction
  tl_plot_xgboost_shap_dependence(model,
    feature = "Petal.Length",
    interaction_feature = "Petal.Width")
}

# }
```
