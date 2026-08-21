# Silhouette Analysis Across Multiple k Values

Silhouette Analysis Across Multiple k Values

## Usage

``` r
tidy_silhouette_analysis(
  data,
  max_k = 10,
  method = "kmeans",
  nstart = 25,
  dist_method = "euclidean",
  linkage_method = "average"
)
```

## Arguments

- data:

  A data frame or tibble

- max_k:

  Maximum number of clusters to test (default: 10)

- method:

  Clustering method: "kmeans" (default) or "hclust"

- nstart:

  If kmeans, number of random starts (default: 25)

- dist_method:

  Distance metric (default: "euclidean")

- linkage_method:

  If hclust, linkage method (default: "average")

## Value

A tibble with columns `k` and `avg_sil_width`. The `"optimal_k"`
attribute contains the k with the highest average silhouette width.

## Examples

``` r
# \donttest{
sil_analysis <- tidy_silhouette_analysis(iris[, 1:4], max_k = 6)
# }
```
