# Plot lift chart for a classification model

Plot lift chart for a classification model

## Usage

``` r
tl_plot_lift(model, new_data = NULL, bins = 10, ...)
```

## Arguments

- model:

  A tidylearn classification model object

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- bins:

  Number of bins for grouping predictions (default: 10)

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
iris_bin <- iris[iris$Species != "setosa", ]
iris_bin$Species <- factor(iris_bin$Species)
model <- tl_model(iris_bin, Species ~ ., method = "logistic")
tl_plot_lift(model)

# }
```
