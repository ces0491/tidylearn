# Plot interaction effects

Plot interaction effects

## Usage

``` r
tl_plot_interaction(
  model,
  var1,
  var2,
  n_points = 100,
  fixed_values = NULL,
  confidence = TRUE,
  ...
)
```

## Arguments

- model:

  A tidylearn model object

- var1:

  First variable in the interaction

- var2:

  Second variable in the interaction

- n_points:

  Number of points to use for continuous variables

- fixed_values:

  Named list of values for other variables in the model

- confidence:

  Logical; whether to show confidence intervals

- ...:

  Additional arguments to pass to predict()

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")

# var2 is drawn as a set of lines across the range of var1
tl_plot_interaction(model, var1 = "wt", var2 = "hp")


# Coarser grid, no ribbon
tl_plot_interaction(model, var1 = "wt", var2 = "hp",
  n_points = 20, confidence = FALSE)

# }
```
