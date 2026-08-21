# Plot calibration curve for a classification model

Plot calibration curve for a classification model

## Usage

``` r
tl_plot_calibration(model, new_data = NULL, bins = 10, ...)
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

A ggplot object with calibration curve
