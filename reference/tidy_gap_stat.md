# Tidy Gap Statistic

Compute gap statistic for determining optimal number of clusters

## Usage

``` r
tidy_gap_stat(data, FUN_cluster = NULL, max_k = 10, B = 50, nstart = 25)
```

## Arguments

- data:

  A data frame or tibble

- FUN_cluster:

  Clustering function (default: uses kmeans internally)

- max_k:

  Maximum number of clusters (default: 10)

- B:

  Number of bootstrap samples (default: 50)

- nstart:

  If using kmeans, number of random starts (default: 25)

## Value

A list of class `"tidy_gap"` containing:

- gap_data: tibble with gap statistics for each k

- k_firstSEmax: optimal k via firstSEmax method (most conservative)

- k_globalmax: optimal k via globalmax method

- k_firstmax: optimal k via firstmax method

- recommended_k: recommended k (uses firstSEmax)

- model: the [`clusGap`](https://rdrr.io/pkg/cluster/man/clusGap.html)
  result

## Examples

``` r
# \donttest{
gap <- tidy_gap_stat(iris[, 1:4], max_k = 6, B = 10)
gap$recommended_k
#> [1] 6
# }
```
