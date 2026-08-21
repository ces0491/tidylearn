# Plot MDS Configuration

Visualize MDS results

## Usage

``` r
plot_mds(mds_obj, color_by = NULL, label_points = TRUE, dim_x = 1, dim_y = 2)
```

## Arguments

- mds_obj:

  A tidy_mds object

- color_by:

  Optional grouping to colour points by: either a column name present in
  the MDS configuration, or a vector as long as the data.

- label_points:

  Logical; add point labels? (default: TRUE)

- dim_x:

  Which dimension for x-axis (default: 1)

- dim_y:

  Which dimension for y-axis (default: 2)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html)
object.

## Examples

``` r
# \donttest{
mds <- tidy_mds(USArrests, method = "classical")
plot_mds(mds)

# }
```
