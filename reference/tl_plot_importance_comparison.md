# Plot feature importance across multiple models

Each model's importance is rescaled on its own: its largest value
becomes 100, or, when no value is positive (a forest's permutation
importance can be negative throughout), its largest magnitude becomes
-100. A factor predictor appears once, under its own name, for every
model: the largest of its design columns stands for it where a method
ranks those columns separately (ridge, lasso, elastic net and xgboost).
A predictor a model was given but did not use scores zero for that
model; one it was never given has no bar for it. Tree, forest and boost
models are given the variables of an interaction such as `wt:hp` rather
than the interaction itself, so it has no bar for them. Features are
ranked on their mean importance over the models that were given them.

## Usage

``` r
tl_plot_importance_comparison(..., top_n = 10, names = NULL)
```

## Arguments

- ...:

  tidylearn model objects to compare

- top_n:

  Number of top features to display (default: 10)

- names:

  Optional character vector of model names, one unique name per model

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
m1 <- tl_model(iris, Sepal.Length ~ ., method = "forest")
m2 <- tl_model(iris, Sepal.Length ~ ., method = "boost")
tl_plot_importance_comparison(m1, m2, names = c("Forest", "Boost"))

# }
```
