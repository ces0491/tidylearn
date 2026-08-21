# Formatted model coefficients table

Produces a styled gt table of model coefficients. Supports linear,
polynomial, logistic, ridge, lasso, and elastic net models.

## Usage

``` r
tl_table_coefficients(model, lambda = "1se", digits = 4, ...)
```

## Arguments

- model:

  A tidylearn model object

- lambda:

  For regularised models: "1se" (default) or "min"

- digits:

  Number of decimal places (default: 4)

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_table_coefficients(model)


  


Linear Model Coefficients
```
