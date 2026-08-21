# Plot confusion matrix for a classification model

Plot confusion matrix for a classification model

## Usage

``` r
tl_plot_confusion(model, new_data = NULL, ...)
```

## Arguments

- model:

  A tidylearn classification model object

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- ...:

  Additional arguments

## Value

A ggplot object with confusion matrix
