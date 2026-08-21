# Tidy Principal Component Analysis

Performs PCA on a dataset using tidyverse principles. Returns a tidy
list containing scores, loadings, variance explained, and the original
model.

## Usage

``` r
tidy_pca(data, cols = NULL, scale = TRUE, center = TRUE, method = "prcomp")
```

## Arguments

- data:

  A data frame or tibble

- cols:

  Columns to include in PCA (tidy select syntax). If NULL, uses all
  numeric columns.

- scale:

  Logical; should variables be scaled to unit variance? Default TRUE.

- center:

  Logical; should variables be centered? Default TRUE.

- method:

  Character; "prcomp" (default, recommended) or "princomp"

## Value

A list of class "tidy_pca" containing:

- scores: tibble of PC scores with observation identifiers

- loadings: tibble of variable loadings in long format

- variance: tibble of variance explained by each PC

- model: the original prcomp/princomp object

- settings: list of scale, center, method used

## Examples

``` r
# Basic PCA
pca_result <- tidy_pca(USArrests)


# Access components
pca_result$scores
#> # A tibble: 50 × 5
#>    .obs_id         PC1     PC2     PC3      PC4
#>    <chr>         <dbl>   <dbl>   <dbl>    <dbl>
#>  1 Alabama     -0.976  -1.12    0.440   0.155  
#>  2 Alaska      -1.93   -1.06   -2.02   -0.434  
#>  3 Arizona     -1.75    0.738  -0.0542 -0.826  
#>  4 Arkansas     0.140  -1.11   -0.113  -0.181  
#>  5 California  -2.50    1.53   -0.593  -0.339  
#>  6 Colorado    -1.50    0.978  -1.08    0.00145
#>  7 Connecticut  1.34    1.08    0.637  -0.117  
#>  8 Delaware    -0.0472  0.322   0.711  -0.873  
#>  9 Florida     -2.98   -0.0388  0.571  -0.0953 
#> 10 Georgia     -1.62   -1.27    0.339   1.07   
#> # ℹ 40 more rows
pca_result$loadings
#> # A tibble: 16 × 3
#>    variable component loading
#>    <chr>    <chr>       <dbl>
#>  1 Murder   PC1       -0.536 
#>  2 Murder   PC2       -0.418 
#>  3 Murder   PC3        0.341 
#>  4 Murder   PC4        0.649 
#>  5 Assault  PC1       -0.583 
#>  6 Assault  PC2       -0.188 
#>  7 Assault  PC3        0.268 
#>  8 Assault  PC4       -0.743 
#>  9 UrbanPop PC1       -0.278 
#> 10 UrbanPop PC2        0.873 
#> 11 UrbanPop PC3        0.378 
#> 12 UrbanPop PC4        0.134 
#> 13 Rape     PC1       -0.543 
#> 14 Rape     PC2        0.167 
#> 15 Rape     PC3       -0.818 
#> 16 Rape     PC4        0.0890
pca_result$variance
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.57     2.48         0.620         0.620
#> 2 PC2       0.995    0.990        0.247         0.868
#> 3 PC3       0.597    0.357        0.0891        0.957
#> 4 PC4       0.416    0.173        0.0434        1    
```
