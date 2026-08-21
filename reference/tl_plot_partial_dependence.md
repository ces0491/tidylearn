# Plot partial dependence for tree-based models

Plot partial dependence for tree-based models

## Usage

``` r
tl_plot_partial_dependence(model, var, n.pts = 20, ...)
```

## Arguments

- model:

  A tidylearn tree-based model object

- var:

  Variable name to plot

- n.pts:

  Number of points for continuous variables (default: 20)

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ ., method = "forest")
tl_plot_partial_dependence(model, var = "wt")
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: argument is not numeric or logical: returning NA
#> Warning: Removed 20 rows containing missing values or values outside the scale range
#> (`geom_line()`).
#> Warning: Removed 20 rows containing missing values or values outside the scale range
#> (`geom_point()`).

# }
```
