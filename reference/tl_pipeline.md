# Create a modeling pipeline

Create a modeling pipeline

## Usage

``` r
tl_pipeline(
  data,
  formula,
  preprocessing = NULL,
  models = NULL,
  evaluation = NULL,
  ...
)
```

## Arguments

- data:

  A data frame containing the data

- formula:

  A formula specifying the model

- preprocessing:

  A list of preprocessing steps

- models:

  A list of models to train

- evaluation:

  A list of evaluation criteria

- ...:

  Additional arguments

## Value

A `tidylearn_pipeline` object (S3 list) with components `$formula`,
`$data`, `$preprocessing`, `$models`, `$evaluation`, and `$results`
(initially `NULL`; populated after
[`tl_run_pipeline`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)).

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .,
  models = list(tree = list(method = "tree")))
print(pipe)
#> Tidylearn Pipeline
#> =================
#> Formula: Species ~ . 
#> Data: 150 observations, 5 variables
#> Preprocessing: impute_missing, standardize, dummy_encode 
#> Models: tree 
#> Evaluation:  cv (5 folds)
#> Metrics: accuracy, precision, recall, f1, auc 
#> Best metric: f1 
# }
```
