# Plot model comparison

Plot model comparison

## Usage

``` r
tl_plot_model_comparison(..., new_data = NULL, metrics = NULL, names = NULL)
```

## Arguments

- ...:

  tidylearn model objects to compare

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- metrics:

  Character vector of metrics to compute

- names:

  Optional character vector of model names

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "lasso")
tl_plot_model_comparison(m1, m2, names = c("Linear", "Lasso"))
#> Evaluating on training data. For model validation, provide separate test data.

# }
```
