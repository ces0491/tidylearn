# Augment Data with Hierarchical Cluster Assignments

Add cluster assignments to original data

## Usage

``` r
augment_hclust(hclust_obj, data, k = NULL, h = NULL)
```

## Arguments

- hclust_obj:

  A tidy_hclust object

- data:

  Original data frame

- k:

  Number of clusters (optional)

- h:

  Height at which to cut (optional)

## Value

A tibble containing the original `data` with an additional `cluster`
integer column indicating cluster assignments.

## Examples

``` r
# \donttest{
hc <- tidy_hclust(USArrests, method = "ward.D2")
augmented <- augment_hclust(hc, USArrests, k = 3)
# }
```
