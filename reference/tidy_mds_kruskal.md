# Kruskal's Non-metric MDS

Performs Kruskal's isoMDS

## Usage

``` r
tidy_mds_kruskal(dist_mat, ndim = 2, ...)
```

## Arguments

- dist_mat:

  A distance matrix (dist object)

- ndim:

  Number of dimensions (default: 2)

- ...:

  Additional arguments passed to MASS::isoMDS()

## Value

A list of class `"tidy_mds"` containing:

- config: tibble of MDS coordinates

- stress: Kruskal stress value

- method: `"Kruskal's isoMDS"`

- model: the [`isoMDS`](https://rdrr.io/pkg/MASS/man/isoMDS.html) result

## Examples

``` r
# \donttest{
d <- dist(USArrests)
mds <- tidy_mds_kruskal(d)
# }
```
