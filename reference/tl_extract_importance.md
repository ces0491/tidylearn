# Extract importance from a tree-based model

Extract importance from a tree-based model

## Usage

``` r
tl_extract_importance(model)
```

## Arguments

- model:

  A tidylearn model object

## Value

A data frame with feature importance values, rescaled so the largest is
100. Empty for a tree with no splits.
