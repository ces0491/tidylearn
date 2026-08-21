# Predict with transfer learning model

Predict with transfer learning model

## Usage

``` r
# S3 method for class 'tidylearn_transfer'
predict(object, new_data, ...)
```

## Arguments

- object:

  A tidylearn_transfer model object

- new_data:

  New data for predictions

- ...:

  Additional arguments

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with a
`.pred` column containing predictions.

## Examples

``` r
# \donttest{
model <- tl_transfer_learning(iris, Species ~ .,
  pretrain_method = "pca", supervised_method = "tree")
#> Transfer Learning Workflow
#> ==========================
#> [Phase 1] Unsupervised pre-training with pca...
#> [Phase 2] Supervised learning with tree...
preds <- predict(model, iris[1:5, ])
# }
```
