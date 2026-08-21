# Print method for tidylearn models

Print method for tidylearn models

## Usage

``` r
# S3 method for class 'tidylearn_model'
print(x, ...)
```

## Arguments

- x:

  A tidylearn model object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
print(model)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ wt + hp 
#> 
#> Training observations: 32 
# }
```
