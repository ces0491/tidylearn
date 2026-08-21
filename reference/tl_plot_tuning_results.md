# Plot hyperparameter tuning results

Plot hyperparameter tuning results

## Usage

``` r
tl_plot_tuning_results(
  model,
  top_n = 5,
  param1 = NULL,
  param2 = NULL,
  plot_type = "scatter"
)
```

## Arguments

- model:

  A tidylearn model object with tuning results

- top_n:

  Number of top parameter sets to highlight

- param1:

  First parameter to plot (for 2D grid or scatter plots)

- param2:

  Second parameter to plot (for 2D grid or scatter plots)

- plot_type:

  Type of plot: "scatter", "grid", "parallel", "importance"

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_tune_grid(iris, Species ~ ., method = "tree",
  param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  folds = 2, verbose = FALSE)
tl_plot_tuning_results(model)

# }
```
