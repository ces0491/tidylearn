# Compare models using cross-validation

Each model is refitted on every fold from its formula, method and
fitting arguments. A model a refit would not reproduce is refused: one
built by
[`tl_semisupervised`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
or
[`tl_anomaly_aware`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md),
or one of
[`tl_auto_ml`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)'s
candidates fitted on features it engineered.

## Usage

``` r
tl_compare_cv(data, models, folds = 5, metrics = NULL, ...)
```

## Arguments

- data:

  A data frame containing the training data

- models:

  A named list of supervised tidylearn models, all of them
  classification or all regression. An unnamed model is named
  `Model_<position>`.

- folds:

  Number of cross-validation folds, a whole number between 2 and
  `nrow(data)`. `nrow(data)` leaves each row out in turn, and each fold
  then scores a single prediction. `"accuracy"`, `"mae"`, `"mse"` and
  `"mape"` average to their values over the left-out predictions. The
  average `"rmse"` is the mean absolute error; `"precision"`,
  `"recall"`, `"sensitivity"`, `"specificity"` and `"f1"` are undefined
  on the folds whose one row gives them nothing to divide by; and
  `"rsq"`, `"auc"` and `"pr_auc"` are undefined on every fold. A run
  scoring any of these warns once.

- metrics:

  Character vector of metrics to compute, from those
  [`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  computes for the task. Defaults to
  `c("accuracy", "precision", "recall", "f1", "auc")` for classification
  and `c("rmse", "mae", "rsq", "mape")` for regression.

- ...:

  Arguments passed to
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  for every fold fit. Each model is refitted with the arguments it was
  built with; anything given here overrides them.

## Value

A list with two elements:

- `$fold_metrics`:

  A data frame with columns `metric`, `value`, `fold`, and `model`
  containing per-fold results for every model. A metric undefined on a
  fold – `"auc"` on a fold holding one class – is `NA` there, and so is
  every metric of a fold none of whose rows can be scored, with a
  warning naming the fold.

- `$summary`:

  A data frame with columns `model`, `metric`, `mean_value`, `sd_value`,
  `min_value`, and `max_value` summarizing cross-validation performance
  over the folds with a value. A metric with no value on any fold is
  `NA` throughout.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
cv$summary
#> # A tibble: 8 × 6
#>   model  metric mean_value sd_value min_value max_value
#>   <chr>  <chr>       <dbl>    <dbl>     <dbl>     <dbl>
#> 1 full   mae         2.17     0.769     1.32      2.80 
#> 2 full   mape       11.2      3.65      7.49     14.8  
#> 3 full   rmse        2.64     0.957     1.55      3.30 
#> 4 full   rsq         0.760    0.174     0.581     0.928
#> 5 simple mae         2.57     0.367     2.18      2.91 
#> 6 simple mape       13.8      1.43     12.7      15.4  
#> 7 simple rmse        3.13     0.467     2.60      3.48 
#> 8 simple rsq         0.690    0.139     0.534     0.798
# }
```
