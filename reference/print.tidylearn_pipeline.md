# Print a tidylearn pipeline

Print a tidylearn pipeline

## Usage

``` r
# S3 method for class 'tidylearn_pipeline'
print(x, ...)
```

## Arguments

- x:

  A tidylearn pipeline object

- ...:

  Additional arguments (not used)

## Value

The input pipeline object `x`, returned invisibly.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .)
print(pipe)
#> Tidylearn Pipeline
#> =================
#> Formula: Species ~ . 
#> Data: 150 observations, 5 variables
#> Preprocessing: impute_missing, standardize, dummy_encode 
#> Models: logistic, tree, forest 
#> Evaluation:  cv (5 folds)
#> Metrics: accuracy, precision, recall, f1, auc 
#> Best metric: f1 
# }
```
