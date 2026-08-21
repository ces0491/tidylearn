# Plot precision-recall curve for a classification model

Plot precision-recall curve for a classification model

## Usage

``` r
tl_plot_precision_recall(model, new_data = NULL, ...)
```

## Arguments

- model:

  A tidylearn classification model object

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- ...:

  Additional arguments

## Value

A ggplot object with precision-recall curve
