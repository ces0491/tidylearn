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

  Data for SHAP value calculation (default: NULL, uses training data).
  The response column is not needed.

- n_samples:

  Number of samples to use (default: 100, NULL for all)

- trees_idx:

  Boosting rounds to include, as a run of consecutive rounds counted
  from 1 such as `1:20` (default: NULL, uses every round)

## Value

A data frame with one column of SHAP values per feature (the columns of
the model's design matrix), a `BIAS` column and a `row_id` column giving
the row of the (sampled) data each row explains. For a multiclass model
there is one block of rows per class, told apart by a `class` column.
The columns of `data` whose names are not already taken – the response,
and any factor predictor, whose SHAP columns are named after its levels
– are appended for reference.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- tl_model(mtcars, mpg ~ ., method = "xgboost", nthread = 2)
  shap <- tl_xgboost_shap(model, n_samples = 20)
}
# }
```
