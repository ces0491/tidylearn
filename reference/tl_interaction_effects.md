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

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt * hp, method = "linear")

# How the effect of weight changes across horsepower
effects <- tl_interaction_effects(model, var = "wt", by_var = "hp")
head(effects$effects)
#>         wt hp      fit       se    lower    upper by_value by_label
#> 1 1.513000 52 33.32234 1.241017 30.88995 35.75474       52       Q0
#> 2 1.552505 52 33.05495 1.210429 30.68251 35.42739       52       Q0
#> 3 1.592010 52 32.78756 1.180197 30.47438 35.10075       52       Q0
#> 4 1.631515 52 32.52017 1.150349 30.26549 34.77485       52       Q0
#> 5 1.671020 52 32.25278 1.120916 30.05578 34.44978       52       Q0
#> 6 1.710525 52 31.98539 1.091933 29.84520 34.12558       52       Q0
effects$slopes
#>      by_value by_label     slope     slope_se
#> Q0       52.0       Q0 -6.768521 5.695734e-16
#> Q25      96.5      Q25 -5.529278 4.387243e-16
#> Q50     123.0      Q50 -4.791302 6.071491e-16
#> Q75     180.0      Q75 -3.203958 4.018576e-16
#> Q100    335.0     Q100  1.112505 4.116338e-16

# slopes$slope_se describes the fitted grid, not the sampling
# uncertainty of the marginal effect -- for that, read the coefficient
summary(model$fit)$coefficients["wt:hp", ]
#>     Estimate   Std. Error      t value     Pr(>|t|) 
#> 0.0278481483 0.0074195805 3.7533319407 0.0008108307 
# }
```
