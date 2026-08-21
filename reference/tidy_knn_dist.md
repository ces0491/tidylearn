# Compute k-NN Distances

Calculate distances to k-th nearest neighbor for each point

## Usage

``` r
tidy_knn_dist(data, k = 4, cols = NULL)
```

## Arguments

- data:

  A data frame or matrix

- k:

  Number of nearest neighbors (default: 4)

- cols:

  Columns to include (tidy select). If NULL, uses all numeric columns.

## Value

A tibble with columns `.obs_id` (observation identifier), `knn_dist`
(distance to k-th nearest neighbor), and `rank` (rank of the k-NN
distance).

## Examples

``` r
# \donttest{
knn <- tidy_knn_dist(iris[, 1:4], k = 5)
# }
```
