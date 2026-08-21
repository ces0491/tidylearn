# Calculate Cluster Validation Metrics

Comprehensive validation metrics for a clustering result

## Usage

``` r
calc_validation_metrics(clusters, data = NULL, dist_mat = NULL)
```

## Arguments

- clusters:

  Vector of cluster assignments

- data:

  Original data frame (for WSS calculation)

- dist_mat:

  Distance matrix (for silhouette)

## Value

A single-row tibble with columns `k`, `min_size`, `max_size`,
`avg_size`, and optionally `avg_silhouette`, `min_silhouette` (if
`dist_mat` provided), and `total_wss` (if `data` provided).

## Examples

``` r
# \donttest{
km <- kmeans(iris[, 1:4], centers = 3, nstart = 25)
d <- dist(iris[, 1:4])
metrics <- calc_validation_metrics(km$cluster, iris[, 1:4], d)
# }
```
