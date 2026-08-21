# Stratified Features via Clustering

Create cluster-specific supervised models for heterogeneous data

## Usage

``` r
tl_stratified_models(
  data,
  formula,
  cluster_method = "kmeans",
  k = 3,
  supervised_method = "linear",
  ...
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula

- cluster_method:

  Clustering method

- k:

  Number of clusters

- supervised_method:

  Supervised learning method

- ...:

  Additional arguments

## Value

A list with class `"tidylearn_stratified"` containing:

- cluster_model:

  The fitted clustering model.

- supervised_models:

  Named list of tidylearn models, one per cluster.

- formula:

  The model formula.

- data:

  The original training data.

## Examples

``` r
# \donttest{
models <- tl_stratified_models(mtcars, mpg ~ ., cluster_method = "kmeans",
                                k = 3, supervised_method = "linear")
#> Note: Response 'mpg' has 8 unique numeric values. Treating as regression. Convert to factor for classification.
#> Note: Response 'mpg' has 6 unique numeric values. Treating as regression. Convert to factor for classification.
# }
```
