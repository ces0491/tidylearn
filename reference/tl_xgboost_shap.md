# Generate SHAP values for XGBoost model interpretation

Generate SHAP values for XGBoost model interpretation

## Usage

``` r
tl_xgboost_shap(model, data = NULL, n_samples = 100, trees_idx = NULL)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- data:

  Data for SHAP value calculation (default: NULL, uses training data)

- n_samples:

  Number of samples to use (default: 100, NULL for all)

- trees_idx:

  Trees to include (default: NULL, uses all trees)

## Value

A data frame with one column of SHAP values per feature, a `BIAS`
column, a `row_id` column, and the original data columns appended for
reference.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost")
  shap <- tl_xgboost_shap(model, n_samples = 20)
}
# }
```
