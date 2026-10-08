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
object. Its data has a `var_value` column and the mean prediction over
the model's training rows, `y`, at each value. For classification, `y`
is a mean class probability and a `class` column says which class: the
positive class (the second level) alone for a two-class model, and every
class, one line each, for more.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ ., method = "forest")
tl_plot_partial_dependence(model, var = "wt")

# }
```
