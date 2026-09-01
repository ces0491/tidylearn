# Plot XGBoost tree visualization

Plot XGBoost tree visualization

## Usage

``` r
tl_plot_xgboost_tree(model, tree_index = 0, ...)
```

## Arguments

- model:

  A tidylearn XGBoost model object

- tree_index:

  Index of the tree to plot (default: 0, first tree)

- ...:

  Additional arguments

## Value

The return value of
[`xgb.plot.tree`](https://rdrr.io/pkg/xgboost/man/xgb.plot.tree.html), a
tree diagram rendered via the DiagrammeR package.

## Examples

``` r
# \donttest{
if (requireNamespace("xgboost", quietly = TRUE) &&
    requireNamespace("DiagrammeR", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "xgboost", nrounds = 10)

  # tree_index is zero-based, so this is the first tree
  tl_plot_xgboost_tree(model, tree_index = 0)
}
#> Warning: Passed unrecognized parameters: tree_index. This warning will become an error in a future version.

{"x":{"diagram":"digraph {\n    graph [ rankdir=TB ]\n\n    0 [ label=\"Petal.Length<3\" ]\n    0 -> 1 [label=\"yes\" color=\"#FF0000\"]\n    0 -> 2 [label=\"no, missing\" color=\"#0000FF\"]\n\n    1 [ label=\"leaf=0.430622011\" ]\n\n    2 [ label=\"leaf=-0.220048919\" ]\n}","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}# }
```
