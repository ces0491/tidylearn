# Create PCA Biplot

Visualize both observations and variables in PC space

## Usage

``` r
tidy_pca_biplot(
  pca_obj,
  pc_x = 1,
  pc_y = 2,
  color_by = NULL,
  arrow_scale = 1,
  label_obs = FALSE,
  label_vars = TRUE
)
```

## Arguments

- pca_obj:

  A tidy_pca object

- pc_x:

  Principal component for x-axis (default: 1)

- pc_y:

  Principal component for y-axis (default: 2)

- color_by:

  Optional grouping to colour points by: either a column name present in
  the PCA scores, or a vector as long as the data.

- arrow_scale:

  Scaling factor for variable arrows (default: 1)

- label_obs:

  Logical; label observations? (default: FALSE)

- label_vars:

  Logical; label variables? (default: TRUE)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
tidy_pca_biplot(pca)

# }
```
