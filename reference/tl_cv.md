# Cross-validation for tidylearn models

Cross-validation for tidylearn models

## Usage

``` r
tl_cv(data, formula, method, folds = 5, metrics = NULL, ...)
```

## Arguments

- data:

  Data frame

- formula:

  Model formula

- method:

  Modeling method

- folds:

  Number of cross-validation folds

- metrics:

  Character vector of metrics to compute on each fold, passed to
  [`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md).
  If `NULL` (the default), `tl_evaluate`'s per-task defaults are used.

- ...:

  Additional arguments passed to
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)

## Value

A list with two elements:

- `$folds`:

  A list of per-fold evaluation
  [tibble](https://tibble.tidyverse.org/reference/tibble.html)s, each
  with `metric` and `value` columns.

- `$summary`:

  A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
  columns `metric`, `mean`, and `sd` summarizing performance across
  folds.

## Examples

``` r
# \donttest{
cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)
cv$summary
#> # A tibble: 3 × 3
#>   metric  mean    sd
#>   <chr>  <dbl> <dbl>
#> 1 mae    2.24  0.333
#> 2 rmse   2.79  0.308
#> 3 rsq    0.613 0.230
# }
```
