# Create interactive visualization dashboard for a model

Create interactive visualization dashboard for a model

## Usage

``` r
tl_dashboard(model, new_data = NULL, ...)
```

## Arguments

- model:

  A tidylearn model object

- new_data:

  Optional data frame for evaluation (if NULL, uses training data)

- ...:

  Additional arguments

## Value

A [`shinyApp`](https://rdrr.io/pkg/shiny/man/shinyApp.html) object.

## Examples

``` r
# \donttest{
if (requireNamespace("shiny")) {
  model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
  app <- tl_dashboard(model)
}
#> Loading required namespace: shiny
# }
```
