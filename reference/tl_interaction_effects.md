# Calculate partial effects based on a model with interactions

Calculate partial effects based on a model with interactions

## Usage

``` r
tl_interaction_effects(model, var, by_var, at_values = NULL, intervals = TRUE)
```

## Arguments

- model:

  A tidylearn model object

- var:

  Variable to calculate effects for

- by_var:

  Variable to calculate effects by (interaction variable)

- at_values:

  Named list of values at which to hold other variables

- intervals:

  Logical; whether to include confidence intervals

## Value

For numeric `var`: a list with `effects` (data frame of predicted values
across the variable range for each level of `by_var`) and `slopes` (data
frame with the slope of `var` at each level of `by_var`). For
categorical `var`: a data frame of predicted values at each factor level
for each level of `by_var`.

`slopes$slope_se` is the standard error of a straight line fitted to the
prediction grid, not the sampling uncertainty of the marginal effect.
For a linear model the grid is exactly linear in `var`, so this is near
zero by construction and should not be read as a precise estimate. Use
`summary(model$fit)` for inference on the interaction coefficient
itself.
