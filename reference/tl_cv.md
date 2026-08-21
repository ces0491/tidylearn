# Cross-validation for tidylearn models

Cross-validation for tidylearn models

## Usage

``` r
tl_cv(data, formula, method, folds = 5, metrics = NULL, transform = NULL, ...)
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

- transform:

  Optional function for feature engineering that has to be refitted per
  fold. It is called with the training rows of each fold and must return
  a list with an `apply` function (applied to both the training and
  assessment rows) and, optionally, a `formula` to fit under. Use this
  for anything that learns parameters from the data – PCA rotations,
  cluster centroids, target encodings – since fitting those before the
  split inflates every fold's score.

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
#> 1 mae    2.09  0.779
#> 2 rmse   2.63  0.927
#> 3 rsq    0.737 0.214
# }
```
