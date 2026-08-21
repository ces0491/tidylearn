# Plot Variance Explained (PCA)

Create combined scree plot showing individual and cumulative variance

## Usage

``` r
plot_variance_explained(variance_tbl, threshold = 0.8)
```

## Arguments

- variance_tbl:

  Variance tibble from tidy_pca

- threshold:

  Horizontal line for variance threshold (default: 0.8 for 80%)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
model <- tl_model(iris[, 1:4], method = "pca")
plot_variance_explained(model$fit$variance_explained)

# }
```
