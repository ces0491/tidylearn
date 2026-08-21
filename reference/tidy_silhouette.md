# Tidy Silhouette Analysis

Compute silhouette statistics for cluster validation

## Usage

``` r
tidy_silhouette(clusters, dist_mat)
```

## Arguments

- clusters:

  Vector of cluster assignments

- dist_mat:

  Distance matrix (dist object)

## Value

A list of class "tidy_silhouette" containing:

- silhouette_data: tibble with silhouette values for each observation

- avg_width: average silhouette width

- cluster_avg: average silhouette width by cluster

## Examples

``` r
# \donttest{
km <- kmeans(iris[, 1:4], centers = 3, nstart = 25)
d <- dist(iris[, 1:4])
sil <- tidy_silhouette(km$cluster, d)
# }
```
