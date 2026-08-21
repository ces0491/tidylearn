# Plot SVM tuning results

Plot SVM tuning results

## Usage

``` r
tl_plot_svm_tuning(model, ...)
```

## Arguments

- model:

  A tidylearn SVM model object

- ...:

  Additional arguments

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
if (requireNamespace("e1071", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "svm",
    kernel = "linear", tune = TRUE, tune_folds = 2)
  tl_plot_svm_tuning(model)
}

# }
```
