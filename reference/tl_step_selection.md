# Perform stepwise selection on a linear model

Perform stepwise selection on a linear model

## Usage

``` r
tl_step_selection(
  data,
  formula,
  direction = "backward",
  criterion = "AIC",
  trace = FALSE,
  steps = 1000,
  ...
)
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the initial model

- direction:

  Direction of stepwise selection: "forward", "backward", or "both"

- criterion:

  Criterion for selection: "AIC" or "BIC"

- trace:

  Logical; whether to print progress

- steps:

  Maximum number of steps to take

- ...:

  Additional arguments to pass to step()

## Value

A `tidylearn_model` object of class `tidylearn_linear` wrapping the
selected [`lm`](https://rdrr.io/r/stats/lm.html) model. Access the
underlying model via `$fit` and the selected formula via
`$spec$formula`. `$data` is the data passed in.

## Details

Every candidate model is fitted on the same rows: those with no missing
value in any variable of `formula`. When that leaves out rows a
candidate could otherwise use, a message gives the count, and the fit
records the rows in its `na.action`, as
[`lm()`](https://rdrr.io/r/stats/lm.html) records the rows it drops
itself.

Forward and both-direction selection start from a model holding the
intercept, or none if `formula` removes it with `- 1`, and any
[`offset()`](https://rdrr.io/r/stats/offset.html) terms in `formula`.
Neither is ever dropped.

## Examples

``` r
# \donttest{
model <- tl_step_selection(mtcars, mpg ~ ., direction = "backward")
summary(model)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ wt + qsec + am 
#> 
#> Training observations: 32 
#> 
#> Training Performance:
#> # A tibble: 3 × 2
#>   metric value
#>   <chr>  <dbl>
#> 1 rmse   2.30 
#> 2 mae    1.93 
#> 3 rsq    0.850
# }
```
