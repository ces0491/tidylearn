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
#> Recommended k: 4 (firstSEmax method)
#> 
#> Alternative methods:
#>   firstSEmax: k = 4 (most conservative)
#>   globalmax:  k = 6 (middle ground)
#>   firstmax:   k = 6 (most liberal)
#> 
#> Gap Statistics (first 10):
#> # A tibble: 6 × 5
#>       k  logW E.logW    gap SE.sim
#>   <int> <dbl>  <dbl>  <dbl>  <dbl>
#> 1     1  4.55   4.62 0.0689 0.0242
#> 2     2  3.81   4.21 0.397  0.0196
#> 3     3  3.52   4.01 0.493  0.0182
#> 4     4  3.37   3.92 0.548  0.0192
#> 5     5  3.28   3.83 0.553  0.0162
#> 6     6  3.18   3.75 0.563  0.0217
# }
```
