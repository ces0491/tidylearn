# Plot a supervised tidylearn model

Dispatches to the appropriate plotting function based on model type and
requested plot type.

## Usage

``` r
tl_plot_model(model, type = "auto", ...)
```

## Arguments

- model:

  A tidylearn supervised model object

- type:

  Plot type. For regression: "auto", "actual_predicted", "residuals",
  "diagnostics". For classification: "auto", "confusion", "roc",
  "precision_recall", "calibration", "lift", "gain". "importance" is
  available for tree-based and regularized models.

- ...:

  Additional arguments passed to the underlying plot function

## Value

A ggplot2 object (invisibly for base-graphics plots)
