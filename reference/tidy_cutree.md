# Cut Hierarchical Clustering Tree

Cut dendrogram to obtain cluster assignments

## Usage

``` r
tidy_cutree(hclust_obj, k = NULL, h = NULL)
```

## Arguments

- hclust_obj:

  A tidy_hclust object or hclust object

- k:

  Number of clusters (optional)

- h:

  Height at which to cut (optional)

## Value

A tibble with columns `.obs_id` (observation identifier) and `cluster`
(integer cluster assignment).

## Examples

``` r
# \donttest{
hc <- tidy_hclust(USArrests, method = "ward.D2")
clusters <- tidy_cutree(hc, k = 3)
# }
```
