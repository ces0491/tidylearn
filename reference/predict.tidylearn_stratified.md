# Predict from stratified models

Predict from stratified models

## Usage

``` r
# S3 method for class 'tidylearn_stratified'
predict(object, new_data = NULL, ...)
```

## Arguments

- object:

  A tidylearn_stratified model object

- new_data:

  New data for predictions

- ...:

  Additional arguments

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with a
`.pred` column containing predictions and a `.cluster` column with
cluster assignments.

## Examples

``` r
# \donttest{
models <- tl_stratified_models(mtcars, mpg ~ .,
  cluster_method = "kmeans", k = 2, supervised_method = "linear")
preds <- predict(models)
# }
```
