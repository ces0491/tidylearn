# Model coefficients as a tibble

Returns the coefficients of a fitted tidylearn model as a tibble, with
standard errors, test statistics, p-values and – on request – confidence
intervals. Available for `"linear"`, `"polynomial"`, `"logistic"`,
`"ridge"`, `"lasso"` and `"elastic_net"`. Other methods have no
coefficients; use
[`tl_table_importance`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md)
for those.

## Usage

``` r
tl_coefficients(
  model,
  conf_int = FALSE,
  level = 0.95,
  exponentiate = FALSE,
  lambda = "1se",
  ...
)
```

## Arguments

- model:

  A tidylearn supervised model object from
  [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md).

- conf_int:

  Whether to add `conf_low` and `conf_high` columns (default `FALSE`).
  Not available for regularised methods.

- level:

  Confidence level for the interval (default 0.95). Ignored unless
  `conf_int = TRUE`.

- exponentiate:

  Whether to report `estimate` and the interval on the odds scale rather
  than the log-odds scale (default `FALSE`). Only meaningful for a
  classification model, whose coefficients are log odds. The standard
  error stays on the log-odds scale it was computed on and is renamed
  `std_error_log` to say so.

- lambda:

  For regularised methods: `"1se"` (default), `"min"`, or a numeric
  penalty value.

- ...:

  Additional arguments (currently unused).

## Value

A tibble, one row per model term. For `"linear"`, `"polynomial"` and
`"logistic"`: `term`, `estimate`, `std_error`, `statistic`, `p_value`,
plus `conf_low` and `conf_high` when `conf_int = TRUE`. For regularised
methods: `term`, `estimate` and the `lambda` the estimate came from –
glmnet reports no standard errors, so there is nothing to test or bound.

## Details

Intervals are Wald intervals, computed from the same standard errors as
the `statistic` and `p_value` columns beside them, so the interval and
the p-value always agree about whether zero is excluded. For `"linear"`
and `"polynomial"` that means *t* quantiles on the residual degrees of
freedom, which is exactly what
[`confint`](https://rdrr.io/r/stats/confint.html) returns for an `lm`.
For `"logistic"` it means *z* quantiles, which is what the reported *z*
statistic implies but not what
[`confint()`](https://rdrr.io/r/stats/confint.html) gives – that
profiles the likelihood, which is the better interval when the sample is
small or a class is nearly separated. Call `stats::confint(model$fit)`
when you want it.

A rank-deficient fit – two perfectly collinear predictors, or a factor
level with no observations – cannot estimate every term. Those terms are
returned with an `NA` estimate rather than dropped, so a term named in
the formula never disappears from the output without saying so.

## See also

[`tl_table_coefficients`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
for the same numbers as a formatted table.

## Examples

``` r
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_coefficients(model)
#> # A tibble: 3 × 5
#>   term        estimate std_error statistic  p_value
#>   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)  37.2      1.60        23.3  2.57e-20
#> 2 wt           -3.88     0.633       -6.13 1.12e- 6
#> 3 hp           -0.0318   0.00903     -3.52 1.45e- 3
tl_coefficients(model, conf_int = TRUE)
#> # A tibble: 3 × 7
#>   term        estimate std_error conf_low conf_high statistic  p_value
#>   <chr>          <dbl>     <dbl>    <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)  37.2      1.60     34.0      40.5        23.3  2.57e-20
#> 2 wt           -3.88     0.633    -5.17     -2.58       -6.13 1.12e- 6
#> 3 hp           -0.0318   0.00903  -0.0502   -0.0133     -3.52 1.45e- 3

# Odds ratios from a logistic fit
am_data <- transform(mtcars, am = factor(am))
model <- tl_model(am_data, am ~ wt, method = "logistic")
tl_coefficients(model, conf_int = TRUE, exponentiate = TRUE)
#> # A tibble: 2 × 7
#>   term           estimate std_error_log conf_low conf_high statistic p_value
#>   <chr>             <dbl>         <dbl>    <dbl>     <dbl>     <dbl>   <dbl>
#> 1 (Intercept) 169460.              4.51 24.6       1.17e+9      2.67 0.00759
#> 2 wt               0.0179          1.44  0.00107   2.99e-1     -2.80 0.00509
```
