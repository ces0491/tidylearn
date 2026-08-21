# Plot cross-validation results for a regularized model

Shows the cross-validation error as a function of lambda for ridge,
lasso, or elastic net models fitted with cv.glmnet.

## Usage

``` r
tl_plot_regularization_cv(model, ...)
```

## Arguments

- model:

  A tidylearn regularized model object (ridge, lasso, or elastic_net)

- ...:

  Additional arguments (currently unused)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ ., method = "ridge")
tl_plot_regularization_cv(model)

# }
```
