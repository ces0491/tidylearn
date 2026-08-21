# Classical (Metric) MDS

Performs classical multidimensional scaling using cmdscale()

## Usage

``` r
tidy_mds_classical(dist_mat, ndim = 2, add_rownames = TRUE)
```

## Arguments

- dist_mat:

  A distance matrix (dist object)

- ndim:

  Number of dimensions (default: 2)

- add_rownames:

  Preserve row names from distance matrix (default: TRUE)

## Value

A list of class `"tidy_mds"` containing:

- config: tibble of MDS coordinates

- stress: `NA` (not applicable for classical MDS)

- gof: goodness-of-fit (proportion of variance retained)

- eigenvalues: numeric vector of eigenvalues

- method: `"Classical MDS"`

- model: the [`cmdscale`](https://rdrr.io/r/stats/cmdscale.html) result

## Examples

``` r
# \donttest{
d <- dist(USArrests)
mds <- tidy_mds_classical(d)
print(mds)
#> Tidy MDS Analysis
#> =================
#> 
#> Method: Classical MDS 
#> Dimensions: 2 
#> Observations: 50 
#> Goodness of Fit: 99.34% 
#> 
#> Configuration (first 6 obs):
#> # A tibble: 6 × 3
#>   .obs_id      Dim1   Dim2
#>   <chr>       <dbl>  <dbl>
#> 1 Alabama     -64.8  11.4 
#> 2 Alaska      -92.8  18.0 
#> 3 Arizona    -124.   -8.83
#> 4 Arkansas    -18.3  16.7 
#> 5 California -107.  -22.5 
#> 6 Colorado    -35.0 -13.7 
# }
```
