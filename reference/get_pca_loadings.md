# Get PCA Loadings in Wide Format

Get PCA Loadings in Wide Format

## Usage

``` r
get_pca_loadings(pca_obj, n_components = NULL)
```

## Arguments

- pca_obj:

  A `tidy_pca` object from
  [`tidy_pca`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md),
  or a PCA model from
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md).

- n_components:

  Number of components to include (default: all)

## Value

A tibble with one row per variable and one column per principal
component, containing the loading values.

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
get_pca_loadings(pca, n_components = 2)
#> # A tibble: 4 × 3
#>   variable    PC1    PC2
#>   <chr>     <dbl>  <dbl>
#> 1 Murder   -0.536 -0.418
#> 2 Assault  -0.583 -0.188
#> 3 UrbanPop -0.278  0.873
#> 4 Rape     -0.543  0.167
# }
```
