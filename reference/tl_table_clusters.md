# Formatted cluster summary table

Produces a styled gt table showing cluster sizes and mean feature
values. Supports kmeans, pam, clara, dbscan, and hclust models.

## Usage

``` r
tl_table_clusters(model, k = 3, digits = 2, ...)
```

## Arguments

- model:

  A tidylearn clustering model object

- k:

  For hclust models, the number of clusters to cut (default: 3)

- digits:

  Number of decimal places (default: 2)

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
tl_table_clusters(model)


  


Cluster Summary
```
