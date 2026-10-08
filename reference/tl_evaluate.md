# Evaluate a tidylearn model

Scores a supervised model's predictions against the observed response.

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
  `"rmse"`, `"mse"`, `"mae"`, `"mape"` and `"rsq"`. Any other name is an
  error.

- ...:

  Additional arguments passed to
  [`predict()`](https://rdrr.io/r/stats/predict.html)

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
columns `metric` (character) and `value` (numeric), containing one row
per requested metric, and for `"auc"` on more than two classes an
`auc_<class>` row per class as well. An unsupervised model has no
response to score against and returns the single row
`metric = "completed"`, `value = 1`.

## Details

The observed response is the formula's left-hand side evaluated on the
scored rows, so a model of `log(mpg)` is scored against `log(mpg)`, the
scale it predicts on. For classification, the observed classes are read
against the classes the model was trained on, whose second is the
positive class; rows of a class the model never saw are left out with a
warning. Rows missing the response or a prediction are dropped.
[`tl_calc_classification_metrics`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
describes `"auc"` and `"pr_auc"` when the scored rows hold a single
class or lack one.

With no row left to score – `new_data` has no rows, or every row is
dropped – it is an error of class `tidylearn_no_scored_rows`, naming the
reason. [`tl_cv`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
catches that class and leaves the fold out.

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
