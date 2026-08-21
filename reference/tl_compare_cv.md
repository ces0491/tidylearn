# Compare models using cross-validation

Compare models using cross-validation

## Usage

``` r
tl_compare_cv(data, models, folds = 5, metrics = NULL, ...)
```

## Arguments

- data:

  A data frame containing the training data

- models:

  A list of tidylearn model objects

- folds:

  Number of cross-validation folds

- metrics:

  Character vector of metrics to compute

- ...:

  Additional arguments

## Value

A list with two elements:

- `$fold_metrics`:

  A data frame with columns `metric`, `value`, `fold`, and `model`
  containing per-fold results for every model.

- `$summary`:

  A data frame with columns `model`, `metric`, `mean_value`, `sd_value`,
  `min_value`, and `max_value` summarizing cross-validation performance.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
cv$summary
#> # A tibble: 8 × 6
#>   model  metric mean_value sd_value min_value max_value
#>   <chr>  <chr>       <dbl>    <dbl>     <dbl>     <dbl>
#> 1 full   mae         2.11    0.579      1.48      2.61 
#> 2 full   mape       11.0     2.93       8.61     14.3  
#> 3 full   rmse        2.68    0.563      2.05      3.13 
#> 4 full   rsq         0.786   0.0713     0.705     0.839
#> 5 simple mae         2.57    0.353      2.17      2.84 
#> 6 simple mape       14.0     2.21      12.5      16.5  
#> 7 simple rmse        3.23    0.478      2.68      3.59 
#> 8 simple rsq         0.690   0.0666     0.613     0.734
# }
```
