# Extract importance from a regularized regression model

Extract importance from a regularized regression model

## Usage

``` r
tl_get_importance_regularized(model, lambda = "1se")
```

## Arguments

- model:

  A tidylearn regularized model object

- lambda:

  Which lambda to use ("1se" or "min", default: "1se")

## Value

A data frame with feature importance values
