# Plot Gap Statistic

Plot Gap Statistic

## Usage

``` r
plot_gap_stat(gap_obj, show_methods = FALSE)
```

## Arguments

- gap_obj:

  A tidy_gap object

- show_methods:

  Logical; show all three k selection methods? (default: FALSE)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
gap <- tidy_gap_stat(iris[, 1:4], max_k = 6, B = 10)
plot_gap_stat(gap)

# }
```
