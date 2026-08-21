# Plot Cluster Size Distribution

Create bar plot of cluster sizes

## Usage

``` r
plot_cluster_sizes(clusters, title = "Cluster Size Distribution")
```

## Arguments

- clusters:

  Vector of cluster assignments

- title:

  Plot title (default: "Cluster Size Distribution")

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
clusters <- kmeans(iris[, 1:4], 3)$cluster
plot_cluster_sizes(clusters)

# }
```
