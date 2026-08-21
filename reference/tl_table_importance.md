# Formatted feature importance table

Produces a styled gt table of feature importance with a colour gradient.
Supports tree-based, regularised, and xgboost models.

## Usage

``` r
tl_table_importance(model, top_n = 20, digits = 2, ...)
```

## Arguments

- model:

  A tidylearn model object

- top_n:

  Maximum number of features to display (default: 20)

- digits:

  Number of decimal places (default: 2)

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(iris, Species ~ ., method = "forest")
tl_table_importance(model)


  


Feature Importance
```
