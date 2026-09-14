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

  Logical; whether to add 95\\ `lower` and `upper`. This needs a model
  whose underlying fit is an `lm` or `glm`; for any other fit a message
  is shown and only point estimates are returned. For a `glm` the
  interval is built on the link scale and transformed to the response
  scale.

## Value

For numeric `var`: a list with `effects` (data frame of predicted values
across the variable range for each value of `by_var`) and `slopes` (data
frame with the slope of `var` at each value of `by_var`). For
categorical `var`: a data frame of predicted values at each factor level
for each level of `by_var`. A numeric `by_var` is evaluated at its
quartiles; quartiles that tie are evaluated once, with a label naming
each quartile they stand for, such as `"Q0/Q25"`.

`fit`, `lower`, `upper` and `slope` are on the response scale whatever
`intervals` is set to: predicted probabilities for a logistic model.
`slope` is the slope of a straight line fitted to `fit` across the range
of `var`, so for a non-linear link it is an average rate of change over
that range.

`slopes$slope_se` is the standard error of a straight line fitted to the
prediction grid, not the sampling uncertainty of the marginal effect.
For a linear model the grid is exactly linear in `var`, so this is near
zero by construction and should not be read as a precise estimate. Use
`summary(model$fit)` for inference on the interaction coefficient
itself.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")

# How the effect of weight changes across horsepower
effects <- tl_interaction_effects(model, var = "wt", by_var = "hp")
head(effects$effects)
#>         wt hp      fit    lower    upper by_value by_label
#> 1 1.513000 52 33.32234 30.78023 35.86445       52       Q0
#> 2 1.552505 52 33.05495 30.57550 35.53440       52       Q0
#> 3 1.592010 52 32.78756 30.37004 35.20508       52       Q0
#> 4 1.631515 52 32.52017 30.16379 34.87655       52       Q0
#> 5 1.671020 52 32.25278 29.95669 34.54887       52       Q0
#> 6 1.710525 52 31.98539 29.74867 34.22211       52       Q0
effects$slopes
#>      by_value by_label     slope     slope_se
#> Q0       52.0       Q0 -6.768521 3.505829e-16
#> Q25      96.5      Q25 -5.529278 4.039654e-16
#> Q50     123.0      Q50 -4.791302 3.746703e-16
#> Q75     180.0      Q75 -3.203958 4.635819e-16
#> Q100    335.0     Q100  1.112505 2.313524e-16

# slopes$slope_se describes the fitted grid, not the sampling
# uncertainty of the marginal effect -- for that, read the coefficient
summary(model$fit)$coefficients["wt:hp", ]
#>     Estimate   Std. Error      t value     Pr(>|t|) 
#> 0.0278481483 0.0074195805 3.7533319407 0.0008108307 
# }
```
