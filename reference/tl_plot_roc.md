# Plot ROC curve for a classification model

Plot ROC curve for a classification model

## Usage

``` r
tl_plot_roc(model, new_data = NULL, ...)
```

## Arguments

- model:

  A tidylearn classification model object

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- ...:

  Additional arguments

## Value

A ggplot object with ROC curve
