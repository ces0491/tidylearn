# Create confidence and prediction interval plots

Create confidence and prediction interval plots

## Usage

``` r
tl_plot_intervals(model, new_data = NULL, level = 0.95, ...)
```

## Arguments

- model:

  A tidylearn regression model object

- new_data:

  Optional data frame for prediction (if NULL, uses training data)

- level:

  Confidence level (default: 0.95)

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt, method = "linear")
tl_plot_intervals(model)

# }
```
