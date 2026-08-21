# Create a comprehensive diagnostic dashboard

Create a comprehensive diagnostic dashboard

## Usage

``` r
tl_diagnostic_dashboard(
  model,
  include_influence = TRUE,
  include_assumptions = TRUE,
  include_performance = TRUE,
  arrange_plots = "grid"
)
```

## Arguments

- model:

  A tidylearn model object

- include_influence:

  Logical; whether to include influence diagnostics

- include_assumptions:

  Logical; whether to include assumption checks

- include_performance:

  Logical; whether to include performance metrics

- arrange_plots:

  Layout arrangement (e.g., "grid", "row", "column")

## Value

A [`grid.arrange`](https://rdrr.io/pkg/gridExtra/man/arrangeGrob.html)
object (a [`grob`](https://rdrr.io/r/grid/grid.grob.html)) containing
the arranged diagnostic plots.

## Examples

``` r
# \donttest{
if (requireNamespace("gridExtra")) {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  tl_diagnostic_dashboard(model)
}

#> TableGrob (3 x 3) "arrange": 7 grobs
#>                     z     cells    name           grob
#> residuals_vs_fitted 1 (1-1,1-1) arrange gtable[layout]
#> residual_hist       2 (1-1,2-2) arrange gtable[layout]
#> qq_plot             3 (1-1,3-3) arrange gtable[layout]
#> cook_distance       4 (2-2,1-1) arrange gtable[layout]
#> leverage_plot       5 (2-2,2-2) arrange gtable[layout]
#> assumptions         6 (2-2,3-3) arrange gtable[layout]
#> performance         7 (3-3,1-1) arrange gtable[layout]
# }
```
