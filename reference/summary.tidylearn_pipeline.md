# Summarize a tidylearn pipeline

Summarize a tidylearn pipeline

## Usage

``` r
# S3 method for class 'tidylearn_pipeline'
summary(object, ...)
```

## Arguments

- object:

  A tidylearn pipeline object

- ...:

  Additional arguments (not used)

## Value

The input pipeline `object`, returned invisibly. Called for its side
effect of printing detailed pipeline and model results.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .)
summary(pipe)
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
