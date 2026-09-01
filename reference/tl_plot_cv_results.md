# Plot cross-validation results

Plot cross-validation results

## Usage

``` r
tl_plot_cv_results(cv_results, metrics = NULL)
```

## Arguments

- cv_results:

  Cross-validation results from tl_cv function

- metrics:

  Character vector of metrics to plot (if NULL, plots all metrics)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
cv <- tl_cv(mtcars, mpg ~ wt + hp, method = "linear", folds = 5)
tl_plot_cv_results(cv)


# One metric rather than every one the folds scored
tl_plot_cv_results(cv, metrics = "rmse")

# }
```
