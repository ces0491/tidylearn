# Plot comparison of cross-validation results

Plot comparison of cross-validation results

## Usage

``` r
tl_plot_cv_comparison(cv_results, metrics = NULL)
```

## Arguments

- cv_results:

  Results from tl_compare_cv function

- metrics:

  Character vector of metrics to plot (if NULL, plots all metrics)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) object
showing boxplots of cross-validation metric distributions for each
model.

## Examples

``` r
# \donttest{
m1 <- tl_model(mtcars, mpg ~ wt, method = "linear")
m2 <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
cv <- tl_compare_cv(mtcars, list(simple = m1, full = m2), folds = 3)
tl_plot_cv_comparison(cv)

# }
```
