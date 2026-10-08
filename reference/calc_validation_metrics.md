# Calculate Cluster Validation Metrics

Comprehensive validation metrics for a clustering result

## Usage

``` r
calc_validation_metrics(clusters, data = NULL, dist_mat = NULL)
```

## Arguments

- clusters:

  Vector of cluster assignments: numeric, factor or character. A label
  of 0 marks noise, as
  [`tidy_dbscan`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  reports it: noise points are left out of every measure and counted in
  `n_noise`.

- data:

  Original data frame (for WSS calculation). WSS is taken over its
  numeric columns, so it needs at least one.

- dist_mat:

  Distance matrix (for silhouette)

## Value

A single-row tibble with columns `k`, `min_size`, `max_size`,
`avg_size`, `n_noise`, and optionally `avg_silhouette`, `min_silhouette`
(if `dist_mat` provided; `NA` for a single cluster), and `total_wss` (if
`data` provided).

## Examples

``` r
# \donttest{
km <- kmeans(iris[, 1:4], centers = 3, nstart = 25)
d <- dist(iris[, 1:4])
metrics <- calc_validation_metrics(km$cluster, iris[, 1:4], d)
# }
```
