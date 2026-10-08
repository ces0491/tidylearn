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

  Optional test data for evaluation. If NULL, the models are scored on
  their training data, which they must share: models fitted on different
  data are an error asking for `new_data`. A model fitted on engineered
  features, as
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  builds some of its candidates, is scored on the training data of the
  others.

- names:

  Optional character vector of model names

- digits:

  Number of decimal places (default: 4)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object. Its
source note counts the rows scored, per model when the models scored
different rows.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ ., method = "linear")
m2 <- tl_model(mtcars, mpg ~ ., method = "lasso")
tl_table_comparison(m1, m2, names = c("Linear", "Lasso"))


  


Model Comparison
```
