# Perform statistical comparison of models using cross-validation

Perform statistical comparison of models using cross-validation

## Usage

``` r
tl_test_model_difference(
  cv_results,
  baseline_model = NULL,
  test = "t.test",
  metric = NULL
)
```

## Arguments

- cv_results:

  Results from tl_compare_cv function

- baseline_model:

  Name of the model to use as baseline for comparison

- test:

  Type of statistical test: "t.test" or "wilcox"

- metric:

  Name of the metric to compare

## Value

A data frame with columns `metric`, `model`, `baseline`, `mean_diff`,
`p_value`, and `p_adj` (Holm-adjusted p-value) containing pairwise
statistical comparisons against the baseline model.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
tl_test_model_difference(cv, baseline_model = "simple", metric = "rmse")
#>   metric model baseline  mean_diff    p_value      p_adj
#> 1   rmse  full   simple -0.5551366 0.03000895 0.03000895
# }
```
