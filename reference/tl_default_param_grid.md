# Create pre-defined parameter grids for common models

Create pre-defined parameter grids for common models

## Usage

``` r
tl_default_param_grid(method, size = "medium", is_classification = TRUE)
```

## Arguments

- method:

  Model method ("tree", "forest", "boost", "svm", etc.)

- size:

  Grid size: "small", "medium", "large"

- is_classification:

  Whether the task is classification or regression

## Value

A named list of parameter values suitable for passing to
[`tl_tune_grid`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
or
[`tl_tune_random`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md).
Each element is a numeric or character vector of candidate values for
that hyperparameter.

## Examples

``` r
# \donttest{
grid <- tl_default_param_grid("tree", size = "small")
grid <- tl_default_param_grid("forest", size = "medium")
# }
```
