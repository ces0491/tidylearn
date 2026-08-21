# Print Method for tidy_pam

Print Method for tidy_pam

## Usage

``` r
# S3 method for class 'tidy_pam'
print(x, ...)
```

## Arguments

- x:

  A tidy_pam object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
pm <- tidy_pam(iris[, 1:4], k = 3)
print(pm)
#> Tidy PAM Clustering
#> ===================
#> 
#> Number of clusters: 3 
#> Average silhouette width: 0.5528 
#> 
#> Medoids:
#> # A tibble: 3 × 6
#>   cluster medoid_index Sepal.Length Sepal.Width Petal.Length Petal.Width
#>     <int>        <int>        <dbl>       <dbl>        <dbl>       <dbl>
#> 1       1            8          5           3.4          1.5         0.2
#> 2       2           79          6           2.9          4.5         1.5
#> 3       3          113          6.8         3            5.5         2.1
#> 
#> Cluster sizes:
#> 
#>  1  2  3 
#> 50 62 38 
# }
```
