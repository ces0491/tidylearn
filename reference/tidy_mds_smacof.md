# SMACOF MDS (Metric or Non-metric)

Performs MDS using SMACOF algorithm from the smacof package

## Usage

``` r
tidy_mds_smacof(dist_mat, ndim = 2, type = "ratio", ...)
```

## Arguments

- dist_mat:

  A distance matrix (dist object)

- ndim:

  Number of dimensions (default: 2)

- type:

  Character; "ratio" for metric, "ordinal" for non-metric (default:
  "ratio")

- ...:

  Additional arguments passed to smacof::mds()

## Value

A list of class `"tidy_mds"` containing:

- config: tibble of MDS coordinates

- stress: stress value from the SMACOF algorithm

- method: character string describing the MDS type

- model: the [`mds`](https://rdrr.io/pkg/smacof/man/smacofSym.html)
  result

## Examples

``` r
# \donttest{
d <- dist(USArrests)
mds <- tidy_mds_smacof(d, type = "ratio")
# }
```
