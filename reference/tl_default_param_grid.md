# Create pre-defined parameter grids for common models

Create pre-defined parameter grids for common models

## Usage

``` r
tl_default_param_grid(method, size = "medium", is_classification = TRUE)
```

## Arguments

- method:

  Model method ("tree", "forest", "boost", "svm", "xgboost", etc.)

- size:

  Grid size: "small", "medium", "large"

- is_classification:

  Whether the grid is for a classification task (the default) or a
  regression one. For regression, an `"svm"` grid also tunes `epsilon`,
  the width of the band within which e1071's regression SVM ignores
  errors, and the large `"forest"` grid centres `nodesize` on
  randomForest's regression default of 5 rather than its classification
  default of 1. The other grids are the same for both tasks.

## Value

A named list of parameter values suitable for passing to
[`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
or
[`tl_tune_random`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md).
Each element is a numeric or character vector of candidate values for
that hyperparameter, or for `"deep"`'s `hidden_layers` a list of
layer-size vectors. The grid is built without the data, so a `"forest"`
`mtry` can exceed the number of predictors; the tuners cap it.
`"polynomial"` tunes `degree`. The `"xgboost"` grids draw on the values
[`tl_tune_xgboost`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
searches by default, and add `nrounds`, which that function chooses by
early stopping and
[`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
has to tune. `"linear"` and `"logistic"` have no tuneable hyperparameter
and return an empty list with a warning, as does an unknown method.

## Examples

``` r
# \donttest{
grid <- tl_default_param_grid("tree", size = "small")
grid <- tl_default_param_grid("forest", size = "medium")
grid <- tl_default_param_grid("svm", is_classification = FALSE)
# }
```
