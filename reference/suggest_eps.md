# Suggest eps Parameter for DBSCAN

Use k-NN distance plot to suggest eps value

## Usage

``` r
suggest_eps(data, minPts = 5, method = "percentile", percentile = 0.95)
```

## Arguments

- data:

  A data frame or matrix

- minPts:

  Minimum points parameter (used as k for k-NN)

- method:

  Method to suggest eps: "percentile" (default), "knee"

- percentile:

  If method="percentile", which percentile to use (default: 0.95)

## Value

A list containing:

- eps: suggested epsilon value

- knn_distances: full tibble of k-NN distances

- method: method used

## Examples

``` r
eps_info <- suggest_eps(iris, minPts = 5)
eps_info$eps
#> [1] 0.75757
```
