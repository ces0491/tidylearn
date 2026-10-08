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

## Details

A parameter whose candidates are not single values, such as
`hidden_layers = list(c(10), c(20, 10))` or a `parms` list, is drawn as
a categorical one, each value labelled as the verbose messages print it.
The importance of a numeric parameter is the absolute correlation of its
values with the score; that of a categorical one is eta squared from a
one-way ANOVA of the score, and 0 when the sets that were scored all
share one value.

## Examples

``` r
# \donttest{
model <- tl_tune_grid(iris, Species ~ ., method = "tree",
  param_grid = list(cp = c(0.01, 0.1), minsplit = c(10, 20)),
  folds = 2, verbose = FALSE)
tl_plot_tuning_results(model)

# }
```
