# Plot a decision tree

Plot a decision tree

## Usage

``` r
tl_plot_tree(model, ...)
```

## Arguments

- model:

  A tidylearn tree model object

- ...:

  Additional arguments to pass to rpart.plot()

## Value

The return value of
[`rpart.plot`](https://rdrr.io/pkg/rpart.plot/man/rpart.plot.html),
called for its side effect of drawing the tree.

## Examples

``` r
# \donttest{
model <- tl_model(iris, Species ~ ., method = "tree")
tl_plot_tree(model)
#> Warning: Cannot retrieve the data used to build the model (so cannot determine roundint and is.binary for the variables).
#> To silence this warning:
#>     Call rpart.plot with roundint=FALSE,
#>     or rebuild the rpart model with model=TRUE.

# }
```
