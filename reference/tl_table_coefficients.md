# Formatted model coefficients table

Produces a styled gt table of model coefficients. Supports linear,
polynomial, logistic, ridge, lasso, and elastic net models. The numbers
come from
[`tl_coefficients`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md),
which returns them as a tibble if you would rather format them yourself.

## Usage

``` r
tl_table_coefficients(
  model,
  lambda = "1se",
  digits = 4,
  conf_int = FALSE,
  level = 0.95,
  exponentiate = FALSE,
  ...
)
```

## Arguments

- model:

  A tidylearn model object

- lambda:

  For regularised models: "1se" (default) or "min"

- digits:

  Number of decimal places (default: 4)

- conf_int:

  Whether to add a confidence interval (default: `FALSE`). Not available
  for regularised models.

- level:

  Confidence level for the interval (default: 0.95)

- exponentiate:

  Whether to report odds ratios rather than log odds (default: `FALSE`).
  Classification models only.

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## See also

[`tl_coefficients`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
for the underlying tibble.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_table_coefficients(model)


  


Linear Model Coefficients
```
