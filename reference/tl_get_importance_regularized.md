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

  Which lambda to use: "1se" (default), "min", or a numeric penalty
  within the fitted path

## Value

A data frame with feature importance values: each coefficient's absolute
value times its predictor's standard deviation, so the ranking does not
depend on units, rescaled to a maximum of 100. For a multiclass model a
predictor takes its largest value across classes.
