# Formatted PCA loadings table

Produces a styled gt table of variable loadings on each principal
component, with a diverging colour scale to highlight strong loadings.

## Usage

``` r
tl_table_loadings(model, n_components = NULL, digits = 3, ...)
```

## Arguments

- model:

  A tidylearn PCA model object

- n_components:

  Number of components to show (default: all)

- digits:

  Number of decimal places (default: 3)

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(iris[, 1:4], method = "pca")
tl_table_loadings(model)


  


PCA Loadings
```
