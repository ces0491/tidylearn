# Compare multiple models in a formatted table

Evaluates multiple tidylearn models and presents the results
side-by-side in a styled gt table.

## Usage

``` r
tl_table_comparison(..., new_data = NULL, names = NULL, digits = 4)
```

## Arguments

- ...:

  tidylearn model objects to compare

- new_data:

  Optional test data for evaluation. If NULL, uses the training data of
  the first model.

- names:

  Optional character vector of model names

- digits:

  Number of decimal places (default: 4)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ ., method = "linear")
m2 <- tl_model(mtcars, mpg ~ ., method = "lasso")
tl_table_comparison(m1, m2, names = c("Linear", "Lasso"))


  


Model Comparison
```
