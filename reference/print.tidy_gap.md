# Print Method for tidy_gap

Print Method for tidy_gap

## Usage

``` r
# S3 method for class 'tidy_gap'
print(x, ...)
```

## Arguments

- x:

  A tidy_gap object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
gap <- tidy_gap_stat(iris[, 1:4], max_k = 6, B = 10)
print(gap)
#> Tidy Gap Statistic
#> ==================
#> 
#> Recommended k: 5 (firstSEmax method)
#> 
#> Alternative methods:
#>   firstSEmax: k = 5 (most conservative)
#>   globalmax:  k = 6 (middle ground)
#>   firstmax:   k = 6 (most liberal)
#> 
#> Gap Statistics (first 10):
#> # A tibble: 6 × 5
#>       k  logW E.logW    gap SE.sim
#>   <int> <dbl>  <dbl>  <dbl>  <dbl>
#> 1     1  4.55   4.64 0.0861 0.0373
#> 2     2  3.81   4.18 0.374  0.0339
#> 3     3  3.52   4.00 0.481  0.0242
#> 4     4  3.37   3.90 0.533  0.0218
#> 5     5  3.28   3.82 0.545  0.0209
#> 6     6  3.18   3.74 0.560  0.0167
# }
```
