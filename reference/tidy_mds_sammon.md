# Sammon Mapping

Performs Sammon's non-linear mapping

## Usage

``` r
tidy_mds_sammon(dist_mat, ndim = 2, ...)
```

## Arguments

- dist_mat:

  A distance matrix (dist object)

- ndim:

  Number of dimensions (default: 2)

- ...:

  Additional arguments passed to MASS::sammon()

## Value

A list of class `"tidy_mds"` containing:

- config: tibble of MDS coordinates

- stress: Sammon stress value

- method: `"Sammon Mapping"`

- model: the [`sammon`](https://rdrr.io/pkg/MASS/man/sammon.html) result

## Examples

``` r
# \donttest{
d <- dist(USArrests)
mds <- tidy_mds_sammon(d)
# }
```
