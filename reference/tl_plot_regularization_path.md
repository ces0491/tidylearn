# Plot regularization path for a regularized model

Plot regularization path for a regularized model

## Usage

``` r
tl_plot_regularization_path(model, label_n = 5, ...)
```

## Arguments

- model:

  A tidylearn regularized model object

- label_n:

  Number of top features to label (default: 5)

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ ., method = "lasso")
tl_plot_regularization_path(model)

# }
```
