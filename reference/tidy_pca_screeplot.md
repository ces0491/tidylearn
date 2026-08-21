# Create PCA Scree Plot

Visualize variance explained by each principal component

## Usage

``` r
tidy_pca_screeplot(pca_obj, type = "proportion", add_line = TRUE)
```

## Arguments

- pca_obj:

  A tidy_pca object

- type:

  Character; "variance" or "proportion" (default)

- add_line:

  Logical; add horizontal line at eigenvalue = 1? (for Kaiser criterion)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
tidy_pca_screeplot(pca)

# }
```
