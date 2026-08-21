# Plot deep learning model training history

Plot deep learning model training history

## Usage

``` r
tl_plot_deep_history(model, metrics = c("loss", "val_loss"), ...)
```

## Arguments

- model:

  A tidylearn deep learning model object

- metrics:

  Which metrics to plot (default: c("loss", "val_loss"))

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
if (FALSE) { # \dontrun{
if (requireNamespace("keras", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 5)
  tl_plot_deep_history(model)
}
} # }
```
