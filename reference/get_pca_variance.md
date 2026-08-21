# Get Variance Explained Summary

Get Variance Explained Summary

## Usage

``` r
get_pca_variance(pca_obj)
```

## Arguments

- pca_obj:

  A `tidy_pca` object from
  [`tidy_pca`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md),
  or a PCA model from
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md).

## Value

A tibble with columns `component`, `sdev`, `variance`, `prop_variance`,
and `cum_variance`.

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
get_pca_variance(pca)
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.57     2.48         0.620         0.620
#> 2 PC2       0.995    0.990        0.247         0.868
#> 3 PC3       0.597    0.357        0.0891        0.957
#> 4 PC4       0.416    0.173        0.0434        1    

# The same accessor works on a tl_model() PCA fit
get_pca_variance(tl_model(USArrests, method = "pca"))
#> # A tibble: 4 × 5
#>   component  sdev variance prop_variance cum_variance
#>   <chr>     <dbl>    <dbl>         <dbl>        <dbl>
#> 1 PC1       1.57     2.48         0.620         0.620
#> 2 PC2       0.995    0.990        0.247         0.868
#> 3 PC3       0.597    0.357        0.0891        0.957
#> 4 PC4       0.416    0.173        0.0434        1    
# }
```
