# Check model assumptions

Check model assumptions

## Usage

``` r
tl_check_assumptions(model, test = TRUE, verbose = TRUE)
```

## Arguments

- model:

  A tidylearn model object

- test:

  Logical; whether to perform statistical tests

- verbose:

  Logical; whether to print test results and explanations

## Value

A named list with one element per assumption checked (`linearity`,
`independence`, `homoscedasticity`, `normality`, `multicollinearity`,
`outliers`), each containing `assumption` (character label), `check`
(logical or `NULL`), `details` (character), and `recommendation`
(character). An additional `overall` element summarises the number of
assumptions checked, violated, and satisfied.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
tl_check_assumptions(model)
#> Registered S3 method overwritten by 'car':
#>   method           from
#>   na.action.merMod lme4
#> Model Assumptions Check Summary:
#> --------------------------------
#> Linearity: SATISFIED
#>   Details: Correlation between fitted values and residuals: 0
#>   Recommendation: Linearity assumption appears satisfied
#> Independence: VIOLATED
#>   Details: Durbin-Watson statistic: 1.3624
#>   Recommendation: Possible autocorrelation in residuals. Check for time-series structure or clustering.
#> Homoscedasticity: SATISFIED
#>   Details: Breusch-Pagan test p-value: 0.6438
#>   Recommendation: Homoscedasticity assumption appears satisfied
#> Normality of Residuals: VIOLATED
#>   Details: Shapiro-Wilk test p-value: 0.0343
#>   Recommendation: Residuals may not be normally distributed. Consider transformations or robust regression.
#> No Multicollinearity: SATISFIED
#>   Details: Maximum VIF: 1.7666
#>   Recommendation: No serious multicollinearity detected
#> No Influential Outliers: VIOLATED
#>   Details: 6 influential observations detected
#>   Recommendation: Consider inspecting observations: 16, 17, 18, 20, 29 ... (and potentially others)
#> $linearity
#> $linearity$assumption
#> [1] "Linearity"
#> 
#> $linearity$check
#> [1] TRUE
#> 
#> $linearity$details
#> [1] "Correlation between fitted values and residuals: 0"
#> 
#> $linearity$recommendation
#> [1] "Linearity assumption appears satisfied"
#> 
#> 
#> $independence
#> $independence$assumption
#> [1] "Independence"
#> 
#> $independence$check
#> [1] FALSE
#> 
#> $independence$details
#> [1] "Durbin-Watson statistic: 1.3624"
#> 
#> $independence$recommendation
#> [1] "Possible autocorrelation in residuals. Check for time-series structure or clustering."
#> 
#> 
#> $homoscedasticity
#> $homoscedasticity$assumption
#> [1] "Homoscedasticity"
#> 
#> $homoscedasticity$check
#>   BP 
#> TRUE 
#> 
#> $homoscedasticity$details
#> [1] "Breusch-Pagan test p-value: 0.6438"
#> 
#> $homoscedasticity$recommendation
#> [1] "Homoscedasticity assumption appears satisfied"
#> 
#> 
#> $normality
#> $normality$assumption
#> [1] "Normality of Residuals"
#> 
#> $normality$check
#> [1] FALSE
#> 
#> $normality$details
#> [1] "Shapiro-Wilk test p-value: 0.0343"
#> 
#> $normality$recommendation
#> [1] "Residuals may not be normally distributed. Consider transformations or robust regression."
#> 
#> 
#> $multicollinearity
#> $multicollinearity$assumption
#> [1] "No Multicollinearity"
#> 
#> $multicollinearity$check
#> [1] TRUE
#> 
#> $multicollinearity$details
#> [1] "Maximum VIF: 1.7666"
#> 
#> $multicollinearity$recommendation
#> [1] "No serious multicollinearity detected"
#> 
#> 
#> $outliers
#> $outliers$assumption
#> [1] "No Influential Outliers"
#> 
#> $outliers$check
#> [1] FALSE
#> 
#> $outliers$details
#> [1] "6 influential observations detected"
#> 
#> $outliers$recommendation
#> [1] "Consider inspecting observations: 16, 17, 18, 20, 29 ... (and potentially others)"
#> 
#> 
#> $overall
#> $overall$status
#> [1] "3 assumption(s) appear to be violated. See details."
#> 
#> $overall$n_checked
#> [1] 6
#> 
#> $overall$n_violated
#> [1] 3
#> 
#> $overall$n_satisfied
#> [1] 3
#> 
#> 
# }
```
