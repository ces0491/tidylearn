# Create Elbow Plot for K-Means

Plot total within-cluster sum of squares vs number of clusters

## Usage

``` r
plot_elbow(wss_data, add_line = FALSE, suggested_k = NULL)
```

## Arguments

- wss_data:

  A tibble with columns k and tot_withinss (from calc_wss)

- add_line:

  Add vertical line at suggested optimal k? (default: FALSE)

- suggested_k:

  If add_line=TRUE, which k to highlight

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
wss <- data.frame(k = 2:6, tot_withinss = c(150, 90, 60, 50, 45))
plot_elbow(wss)

# }
```
