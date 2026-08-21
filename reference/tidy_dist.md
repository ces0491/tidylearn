# Tidy Distance Matrix Computation

Compute distance matrices with tidy output

## Usage

``` r
tidy_dist(data, method = "euclidean", cols = NULL, ...)
```

## Arguments

- data:

  A data frame or tibble

- method:

  Character; distance method (default: "euclidean"). Options:
  "euclidean", "manhattan", "maximum", "gower"

- cols:

  Columns to include (tidy select). If NULL, uses all numeric columns.

- ...:

  Additional arguments passed to distance functions

## Value

A [`dist`](https://rdrr.io/r/stats/dist.html) object containing the
computed distance matrix.

## Examples

``` r
# \donttest{
d <- tidy_dist(iris[, 1:4], method = "euclidean")
# }
```
