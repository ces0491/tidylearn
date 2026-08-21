# Plot feature importance across multiple models

Plot feature importance across multiple models

## Usage

``` r
tl_plot_importance_comparison(..., top_n = 10, names = NULL)
```

## Arguments

- ...:

  tidylearn model objects to compare

- top_n:

  Number of top features to display (default: 10)

- names:

  Optional character vector of model names

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
m1 <- tl_model(iris, Species ~ ., method = "forest")
m2 <- tl_model(iris, Species ~ ., method = "boost")
tl_plot_importance_comparison(m1, m2, names = c("Forest", "Boost"))

# }
```
