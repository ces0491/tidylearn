# Plot Dendrogram with Cluster Highlights

Enhanced dendrogram with colored cluster rectangles

## Usage

``` r
plot_dendrogram(
  hclust_obj,
  k = NULL,
  title = "Hierarchical Clustering Dendrogram"
)
```

## Arguments

- hclust_obj:

  Hierarchical clustering object (hclust or tidy_hclust)

- k:

  Number of clusters to highlight

- title:

  Plot title

## Value

Invisibly returns the [`hclust`](https://rdrr.io/r/stats/hclust.html)
object. The dendrogram is drawn as a side effect.

## Examples

``` r
# \donttest{
hc <- hclust(dist(iris[, 1:4]))
plot_dendrogram(hc, k = 3)

# }
```
