# Compare Distance Methods

Compute distances using multiple methods for comparison

## Usage

``` r
compare_distances(data, methods = c("euclidean", "manhattan", "maximum"))
```

## Arguments

- data:

  A data frame or tibble

- methods:

  Character vector of methods to compare

## Value

A named list of [`dist`](https://rdrr.io/r/stats/dist.html) objects, one
per method.

## Examples

``` r
# \donttest{
dists <- compare_distances(
  iris[, 1:4], methods = c("euclidean", "manhattan")
)
# }
```
