# Print Method for tidy_silhouette

Print Method for tidy_silhouette

## Usage

``` r
# S3 method for class 'tidy_silhouette'
print(x, ...)
```

## Arguments

- x:

  A tidy_silhouette object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
km <- kmeans(iris[, 1:4], centers = 3, nstart = 25)
d <- dist(iris[, 1:4])
sil <- tidy_silhouette(km$cluster, d)
print(sil)
#> Tidy Silhouette Analysis
#> ========================
#> 
#> Average silhouette width: 0.5528 
#> 
#> Interpretation:
#>   > 0.70: Strong structure
#>   > 0.50: Reasonable structure
#>   > 0.25: Weak structure
#>   < 0.25: No substantial structure
#> 
#> By Cluster:
#> # A tibble: 3 × 3
#>   cluster     n avg_sil_width
#>     <dbl> <int>         <dbl>
#> 1       1    50         0.798
#> 2       2    38         0.451
#> 3       3    62         0.417
# }
```
