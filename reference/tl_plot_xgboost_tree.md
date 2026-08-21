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
