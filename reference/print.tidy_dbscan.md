# Print Method for tidy_dbscan

Print Method for tidy_dbscan

## Usage

``` r
# S3 method for class 'tidy_dbscan'
print(x, ...)
```

## Arguments

- x:

  A tidy_dbscan object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
db <- tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)
print(db)
#> Tidy DBSCAN Clustering
#> ======================
#> 
#> Parameters:
#>   eps (neighborhood radius): 0.5 
#>   minPts (minimum points):  5 
#> 
#> Results:
#>   Number of clusters: 2 
#>   Number of noise points: 17 
#>   Proportion noise: 11.3% 
#> 
#> Cluster Summary:
#> # A tibble: 2 × 3
#>   cluster  size n_core
#>     <int> <int>  <int>
#> 1       1    49      0
#> 2       2    84      0
#> 
#> Use augment_dbscan() to add cluster assignments to your data
# }
```
