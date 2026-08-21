# Formatted evaluation metrics table

Produces a styled gt table of model evaluation metrics from
[`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md).

## Usage

``` r
tl_table_metrics(model, new_data = NULL, digits = 4, ...)
```

## Arguments

- model:

  A tidylearn supervised model object

- new_data:

  Optional test data. If NULL, uses training data.

- digits:

  Number of decimal places (default: 4)

- ...:

  Additional arguments passed to `tl_evaluate`

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_table_metrics(model)


  


Model Evaluation Metrics
```
