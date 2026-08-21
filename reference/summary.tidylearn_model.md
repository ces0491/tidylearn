# Summary method for tidylearn models

Summary method for tidylearn models

## Usage

``` r
# S3 method for class 'tidylearn_model'
summary(object, ...)
```

## Arguments

- object:

  A tidylearn model object

- ...:

  Additional arguments (ignored)

## Value

The input `object`, returned invisibly. Called for its side effect of
printing model summary and training performance.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
summary(model)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ wt + hp 
#> 
#> Training observations: 32 
#> 
#> Training Performance:
#> # A tibble: 3 × 2
#>   metric value
#>   <chr>  <dbl>
#> 1 rmse   2.47 
#> 2 mae    1.90 
#> 3 rsq    0.827
# }
```
