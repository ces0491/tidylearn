# Create formatted tables for tidylearn models

Dispatches to the appropriate table function based on model type and
requested table type. Requires the gt package.

## Usage

``` r
tl_table(model, type = "auto", ...)
```

## Arguments

- model:

  A tidylearn model object

- type:

  Table type (default: "auto"). For supervised models: "metrics",
  "coefficients", "confusion", "importance". For unsupervised models:
  "variance", "loadings", "clusters". MDS models are not supported.

- ...:

  Additional arguments passed to the underlying table function

## Value

A [`gt`](https://gt.rstudio.com/reference/gt.html) table object.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_table(model)


  


Model Evaluation Metrics
```
