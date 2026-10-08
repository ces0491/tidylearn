# Visualize Association Rules

Create visualizations of association rules

## Usage

``` r
visualize_rules(rules_obj, method = "scatter", top_n = 50, ...)
```

## Arguments

- rules_obj:

  A tidy_apriori object or an arules rules object. A table of rules is
  refused: the plots need the rules object.

- method:

  Visualization method: "scatter" (default), drawn by tidylearn, or a
  method of arulesViz's
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html), such as
  "graph", "grouped", "matrix" or "paracoord"

- top_n:

  Number of rules to visualize, those with the highest lift (default:
  50)

- ...:

  Additional arguments passed to plot() for rules visualization

## Value

For `method = "scatter"`, a
[`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) object.
Other methods return what arulesViz's
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) returns: a
ggplot object for "graph", "grouped" and "matrix", and for "paracoord",
which draws with grid, a grid `vpPath`.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  visualize_rules(res, method = "scatter")
}

# }
```
