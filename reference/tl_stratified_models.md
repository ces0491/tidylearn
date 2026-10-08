# Stratified Features via Clustering

Create cluster-specific supervised models for heterogeneous data

## Usage

``` r
tl_stratified_models(
  data,
  formula,
  cluster_method = "kmeans",
  k = 3,
  supervised_method = "tree",
  ...,
  cluster_args = list()
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula

- cluster_method:

  Clustering method: `"kmeans"` (default), `"pam"`, `"clara"` or
  `"hclust"`, whose tree is cut at `k`. Only k-means can assign new
  rows, so the others predict their training data alone.

- k:

  Number of clusters

- supervised_method:

  Supervised learning method (default: `"tree"`, which handles both
  regression and classification). `"linear"` needs a numeric response
  and refuses a factor.

- ...:

  Additional arguments for the supervised models

- cluster_args:

  A named list of arguments for the clustering step, such as
  `list(nstart = 5)` for k-means. Pass k as `k`.

## Value

A list with class `"tidylearn_stratified"` containing:

- cluster_model:

  The fitted clustering model.

- clusters:

  The training rows' cluster assignments.

- supervised_models:

  Named list of tidylearn models, one per cluster that holds more than
  one class.

- single_class_clusters:

  Named character vector giving, for each cluster whose rows all hold
  one class, that class.

- formula:

  The model formula.

- data:

  The original training data.

## Details

The rows are clustered on the predictors the formula names, and a model
is fitted to each cluster. A cluster whose rows all hold one class has
nothing for a classifier to separate, so it gets no model and its rows
are predicted as that class.

## Examples

``` r
# \donttest{
models <- tl_stratified_models(mtcars, mpg ~ ., cluster_method = "kmeans",
                                k = 3, supervised_method = "linear")
#> Note: Response 'mpg' has 6 unique numeric values. Treating as regression. Convert to factor for classification.
# }
```
