# Augment Data with K-Means Cluster Assignments

Augment Data with K-Means Cluster Assignments

## Usage

``` r
augment_kmeans(kmeans_obj, data)
```

## Arguments

- kmeans_obj:

  A tidy_kmeans object

- data:

  Original data frame

## Value

A tibble containing the original `data` with an additional `cluster`
factor column indicating cluster assignments.

## Examples

``` r
# \donttest{
km <- tidy_kmeans(iris[, 1:4], k = 3)
augmented <- augment_kmeans(km, iris)
# }
```
