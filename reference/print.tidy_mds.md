# Print Method for tidy_mds

Print Method for tidy_mds

## Usage

``` r
# S3 method for class 'tidy_mds'
print(x, ...)
```

## Arguments

- x:

  A tidy_mds object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
mds <- tidy_mds(USArrests, method = "classical")
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
