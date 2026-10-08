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

  New data for predictions. NULL predicts the training rows from their
  stored cluster assignments, which works for every clustering method;
  new rows can be assigned by k-means only.

- ...:

  Additional arguments passed to each cluster's model, such as `type`

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) of the
columns each cluster's model returns for the requested `type` – `.pred`
by default, one column per class for `type = "prob"` – and a `.cluster`
column with cluster assignments. Rows of a single-class cluster are
predicted as that class, with probability 1.

## Examples

``` r
# \donttest{
models <- tl_stratified_models(mtcars, mpg ~ .,
  cluster_method = "kmeans", k = 2, supervised_method = "linear")
preds <- predict(models)
# }
```
