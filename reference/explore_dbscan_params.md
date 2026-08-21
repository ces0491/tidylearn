# Explore DBSCAN Parameters

Test multiple eps and minPts combinations

## Usage

``` r
explore_dbscan_params(data, eps_values, minPts_values)
```

## Arguments

- data:

  A data frame or matrix

- eps_values:

  Vector of eps values to test

- minPts_values:

  Vector of minPts values to test

## Value

A tibble with columns `eps`, `minPts`, `n_clusters`, `n_noise`, and
`prop_noise` for each parameter combination.

## Examples

``` r
# \donttest{
params <- explore_dbscan_params(iris[, 1:4],
  eps_values = c(0.3, 0.5, 0.8), minPts_values = c(3, 5))
# }
```
