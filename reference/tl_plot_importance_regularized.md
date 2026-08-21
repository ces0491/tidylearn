# Plot variable importance for a regularized model

Plot variable importance for a regularized model

## Usage

``` r
tl_plot_importance_regularized(model, lambda = "1se", top_n = 20, ...)
```

## Arguments

- model:

  A tidylearn regularized model object

- lambda:

  Which lambda to use ("1se" or "min", default: "1se")

- top_n:

  Number of top features to display (default: 20)

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ ., method = "lasso")
tl_plot_importance_regularized(model)

# }
```
