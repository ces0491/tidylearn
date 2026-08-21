# Transfer Learning Workflow

Use unsupervised pre-training (e.g., autoencoder features) before
supervised learning

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

  Model formula

- pretrain_method:

  Pre-training method: "pca", "autoencoder"

- supervised_method:

  Supervised learning method (default: `"tree"`, which handles both
  regression and classification with any number of classes).
  `"logistic"` is binary-only and errors on a response with more than
  two levels.

- ...:

  Additional arguments

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
