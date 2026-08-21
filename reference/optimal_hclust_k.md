# Determine Optimal Number of Clusters for Hierarchical Clustering

Use silhouette or gap statistic to find optimal k

## Usage

``` r
optimal_hclust_k(hclust_obj, method = "silhouette", max_k = 10)
```

## Arguments

- hclust_obj:

  A tidy_hclust object

- method:

  Character; "silhouette" (default) or "gap"

- max_k:

  Maximum number of clusters to test (default: 10)

## Value

A list containing:

- optimal_k: the recommended number of clusters

- method: the evaluation method used

- values: numeric vector of evaluation scores (for silhouette)

- k_range: integer vector of k values tested (for silhouette)

If `method = "gap"`, returns a `tidy_gap` object instead.

## Examples

``` r
# \donttest{
hc <- tidy_hclust(USArrests, method = "ward.D2")
opt <- optimal_hclust_k(hc, method = "silhouette", max_k = 6)
# }
```
