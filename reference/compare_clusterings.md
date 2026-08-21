# Compare Multiple Clustering Results

Compare Multiple Clustering Results

## Usage

``` r
compare_clusterings(cluster_list, data, dist_mat = NULL)
```

## Arguments

- cluster_list:

  Named list of cluster assignment vectors

- data:

  Original data

- dist_mat:

  Distance matrix

## Value

A tibble with one row per clustering method and columns for each
validation metric (see
[`calc_validation_metrics`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)),
plus a `method` column identifying the clustering.

## Examples

``` r
# \donttest{
km3 <- kmeans(iris[, 1:4], 3, nstart = 25)$cluster
km4 <- kmeans(iris[, 1:4], 4, nstart = 25)$cluster
compare_clusterings(list(k3 = km3, k4 = km4), iris[, 1:4])
#> # A tibble: 2 × 8
#>   method     k min_size max_size avg_size avg_silhouette min_silhouette
#>   <chr>  <int>    <int>    <int>    <dbl>          <dbl>          <dbl>
#> 1 k3         3       38       62     50            0.553         0.0264
#> 2 k4         4       28       50     37.5          0.498        -0.0181
#> # ℹ 1 more variable: total_wss <dbl>
# }
```
