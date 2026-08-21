# Print Method for tidy_pca

Print Method for tidy_pca

## Usage

``` r
# S3 method for class 'tidy_pca'
print(x, ...)
```

## Arguments

- x:

  A tidy_pca object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
print(pca)
#> Tidy PCA Analysis
#> =================
#> 
#> Number of observations: 50 
#> Number of variables: 4 
#> Number of components: 4 
#> Settings: scale = TRUE , center = TRUE 
#> 
#> Variance Explained:
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.57     2.48         0.620         0.620
#> 2 PC2       0.995    0.990        0.247         0.868
#> 3 PC3       0.597    0.357        0.0891        0.957
#> 4 PC4       0.416    0.173        0.0434        1    
#> 
#> Access components with:
#>   $scores    - PC scores for each observation
#>   $loadings  - Variable loadings on each PC
#>   $variance  - Variance explained by each PC
#>   $model     - Original PCA model object
# }
```
