# Print EDA results

Print EDA results

## Usage

``` r
# S3 method for class 'tidylearn_eda'
print(x, ...)
```

## Arguments

- x:

  A tidylearn_eda object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
eda <- tl_explore(iris, response = "Species")
#> Running Exploratory Data Analysis...
#> [1/4] PCA analysis...
#> [2/4] Finding optimal clusters...
#> [3/4] Clustering analysis...
#> [4/4] Distance analysis...
#> EDA complete!
print(eda)
#> tidylearn Exploratory Data Analysis
#> ===================================
#> Observations: 150 
#> Variables: 4 
#> Optimal clusters: 2 
#> 
#> PCA Variance Explained (first 5 components):
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.71    2.92         0.730          0.730
#> 2 PC2       0.956   0.914        0.229          0.958
#> 3 PC3       0.383   0.147        0.0367         0.995
#> 4 PC4       0.144   0.0207       0.00518        1    
#> 
#> Cluster sizes (k = 2 ):
#> 
#>  1  2 
#> 53 97 
# }
```
