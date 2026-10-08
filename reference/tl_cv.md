# Cross-validation for tidylearn models

Each fold's model is scored with
[`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md),
so the response is read as that function reads it: a transformed
left-hand side on its own scale, and classes against the fold model's.

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
  for every fold. Arguments holding one value per row of `data` –
  `weights`, `subset`, `offset`, `foldid` and `strata` – are refused,
  since they cannot follow the rows into a fold.

## Value

A list with two elements:

- `$folds`:

  A list of per-fold evaluation
  [tibble](https://tibble.tidyverse.org/reference/tibble.html)s, each
  with `metric` and `value` columns.

- `$summary`:

  A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
  columns `metric`, `mean`, and `sd` summarizing performance across
  folds. A metric undefined on a fold – auc on a fold holding one class
  – is `NA` there and left out of the mean and sd. So is every metric of
  a fold none of whose rows can be scored, with a warning giving the
  reason. A metric with no value on any fold has `NA` mean and sd.

## Examples

``` r
# \donttest{
cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 3)
cv$summary
#> # A tibble: 3 × 3
#>   metric  mean    sd
#>   <chr>  <dbl> <dbl>
#> 1 mae    1.97  0.771
#> 2 rmse   2.45  0.869
#> 3 rsq    0.810 0.133
# }
```
