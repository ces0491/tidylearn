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

  A named list of preprocessing switches, each `TRUE` or `FALSE`:
  `impute_missing` (default `TRUE`) replaces missing predictor values
  with the training median or mode; `standardize` (default `TRUE`)
  centres and scales numeric predictors where that leaves the model the
  formula describes unchanged. It leaves alone any column the formula
  uses inside a function call such as
  [`log()`](https://rdrr.io/r/base/Log.html),
  [`poly()`](https://rdrr.io/r/stats/poly.html) or
  [`offset()`](https://rdrr.io/r/stats/offset.html), every column when
  the formula has no intercept, and the columns of an interaction whose
  lower-order terms are not all in the formula; `dummy_encode` (default
  `TRUE`) only records that categorical predictors are encoded by each
  model's fitting function, and cannot be set to `FALSE`.

- models:

  A list of models to train

- evaluation:

  A list of evaluation criteria

- ...:

  Not used. Anything passed here is an error, so a misspelt argument
  such as `evalution` is reported rather than ignored.

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
