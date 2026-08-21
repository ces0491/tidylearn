# Plot k-NN Distance Plot

Visualize k-NN distances to help choose eps

## Usage

``` r
plot_knn_dist(data, k = 4, add_suggestion = TRUE, percentile = 0.95)
```

## Arguments

- data:

  A data frame or tidy_knn_dist result

- k:

  If data is a data frame, k for k-NN (default: 4)

- add_suggestion:

  Add suggested eps line? (default: TRUE)

- percentile:

  Percentile for suggestion (default: 0.95)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
plot_knn_dist(iris[, 1:4], k = 5)

# }
```
