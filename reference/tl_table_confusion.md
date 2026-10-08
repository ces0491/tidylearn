# Formatted confusion matrix table

Produces a styled gt confusion matrix with correct predictions
highlighted. Only available for classification models.

## Usage

``` r
tl_table_confusion(model, new_data = NULL, ...)
```

## Arguments

- model:

  A tidylearn classification model

- new_data:

  Optional test data. If NULL, uses training data.

- ...:

  Additional arguments (currently unused)

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object, with a
row and a column for each class the model was trained on. Rows of a
class the model never saw are left out, with a warning.

## Examples

``` r
# \donttest{
model <- tl_model(iris, Species ~ ., method = "forest")
tl_table_confusion(model)


  


Confusion Matrix
```
