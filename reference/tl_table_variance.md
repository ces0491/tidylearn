# Formatted PCA variance explained table

Produces a styled gt table of variance explained by each principal
component, with a colour gradient on cumulative variance.

## Usage

``` r
tl_table_variance(model, n_components = NULL, digits = 4, ...)
```

## Arguments

- model:

  A tidylearn PCA model object

- n_components:

  Maximum number of components to show (default: all)

- digits:

  Number of decimal places (default: 4)

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(iris[, 1:4], method = "pca")
tl_table_variance(model)


  


PCA Variance Explained
```
