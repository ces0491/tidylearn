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
#> 1 full   mae         2.01    0.0456     1.98      2.07 
#> 2 full   mape       10.6     2.53       9.12     13.5  
#> 3 full   rmse        2.65    0.123      2.54      2.78 
#> 4 full   rsq         0.781   0.0669     0.705     0.831
#> 5 simple mae         2.45    0.174      2.34      2.65 
#> 6 simple mape       13.3     1.82      11.9      15.4  
#> 7 simple rmse        3.17    0.0705     3.13      3.25 
#> 8 simple rsq         0.691   0.0584     0.627     0.742
# }
```
