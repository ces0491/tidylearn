# Plot Silhouette Analysis

Plot Silhouette Analysis

## Usage

``` r
plot_silhouette(sil_obj)
```

## Arguments

- sil_obj:

  A tidy_silhouette object or tibble from tidy_silhouette_analysis

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
km <- kmeans(iris[, 1:4], centers = 3, nstart = 25)
d <- dist(iris[, 1:4])
sil <- tidy_silhouette(km$cluster, d)
plot_silhouette(sil)

# }
```
