# Plot an unsupervised tidylearn model

Dispatches to the appropriate plotting function based on the
unsupervised model method.

## Usage

``` r
tl_plot_unsupervised(model, type = "auto", ...)
```

## Arguments

- model:

  A tidylearn unsupervised model object

- type:

  Plot type. Each unsupervised method has one plot, so `"auto"` is the
  only value, and anything else is an error.

- ...:

  Additional arguments passed to the underlying plot function

## Value

A ggplot2 object or invisible result
