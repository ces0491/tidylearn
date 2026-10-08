# Transfer Learning Workflow

Use unsupervised pre-training before supervised learning: the predictors
the formula names are projected onto their principal components, and the
supervised model is fitted on the component scores.

## Usage

``` r
tl_transfer_learning(
  data,
  formula,
  pretrain_method = "pca",
  supervised_method = "tree",
  ...
)
```

## Arguments

- data:

  Training data

- formula:

  Model formula, or a string that parses as one. The PCA is fitted on
  the numeric predictors it names.

- pretrain_method:

  Pre-training method. Only `"pca"` is available:
  [`predict()`](https://rdrr.io/r/stats/predict.html) has to project new
  rows, and PCA is the reduction that can.

- supervised_method:

  Supervised learning method (default: `"tree"`, which handles both
  regression and classification with any number of classes).
  `"logistic"` is binary-only and errors on a response with more than
  two levels.

- ...:

  Additional arguments passed to
  [`tl_reduce_dimensions`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md),
  such as `n_components`

## Value

A list with class `"tidylearn_transfer"` containing:

- pretrain_model:

  The fitted dimensionality reduction model.

- supervised_model:

  The fitted supervised tidylearn model.

- formula:

  The model formula.

- method:

  The supervised learning method used.

## Examples

``` r
# \donttest{
model <- tl_transfer_learning(iris, Species ~ .,
  pretrain_method = "pca", supervised_method = "tree")
#> Transfer Learning Workflow
#> ==========================
#> [Phase 1] Unsupervised pre-training with pca...
#> [Phase 2] Supervised learning with tree...
# }
```
