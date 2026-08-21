# Evaluate a tidylearn model

Evaluate a tidylearn model

## Usage

``` r
tl_evaluate(object, new_data = NULL, metrics = NULL, ...)
```

## Arguments

- object:

  A tidylearn model object

- new_data:

  Optional new data for evaluation (if NULL, uses training data)

- metrics:

  Character vector of metrics to compute. If `NULL` (the default),
  `"accuracy"` is used for classification models and
  `c("rmse", "mae", "rsq")` for regression models. Classification
  supports `"accuracy"`, `"precision"`, `"recall"`, `"sensitivity"`,
  `"specificity"`, `"f1"`, `"auc"` and `"pr_auc"`; regression supports
  `"rmse"`, `"mse"`, `"mae"`, `"mape"` and `"rsq"`.

- ...:

  Additional arguments passed to
  [`predict()`](https://rdrr.io/r/stats/predict.html)

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
columns `metric` (character) and `value` (numeric), containing one row
per requested metric.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_evaluate(model)
#> # A tibble: 3 × 2
#>   metric value
#>   <chr>  <dbl>
#> 1 rmse   2.47 
#> 2 mae    1.90 
#> 3 rsq    0.827
tl_evaluate(model, metrics = c("rmse", "mape"))
#> # A tibble: 2 × 2
#>   metric value
#>   <chr>  <dbl>
#> 1 rmse    2.47
#> 2 mape    9.74
# }
```
