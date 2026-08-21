# Print Method for tidy_kmeans

Print Method for tidy_kmeans

## Usage

``` r
# S3 method for class 'tidy_kmeans'
print(x, ...)
```

## Arguments

- x:

  A tidy_kmeans object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
km <- tidy_kmeans(iris[, 1:4], k = 3)
print(km)
#> Tidy K-Means Clustering
#> =======================
#> 
#> Number of clusters: 3 
#> Cluster sizes: 50, 38, 62 
#> Total within-cluster SS: 78.85 
#> Between-cluster SS: 602.52 
#> Iterations: 2 
#> Converged: TRUE 
#> 
#> Cluster Centers:
#> # A tibble: 3 × 5
#>   cluster Sepal.Length Sepal.Width Petal.Length Petal.Width
#>     <int>        <dbl>       <dbl>        <dbl>       <dbl>
#> 1       1         5.01        3.43         1.46       0.246
#> 2       2         6.85        3.07         5.74       2.07 
#> 3       3         5.90        2.75         4.39       1.43 
# }
```
