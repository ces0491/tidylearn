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

  The `minPts` you will pass to
  [`tidy_dbscan`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  (default: 5). The k-NN distance is read at `k = minPts - 1`, the
  neighbours a core point needs besides itself, as
  [`kNNdistplot`](http://michael.hahsler.net/dbscan/reference/kNNdist.md)
  does.

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
#> [1] 0.7179749
```
