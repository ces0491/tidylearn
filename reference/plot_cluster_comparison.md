# Create Cluster Comparison Plot

Compare multiple clustering results side-by-side

## Usage

``` r
plot_cluster_comparison(data, cluster_cols, x_col, y_col)
```

## Arguments

- data:

  Data frame with multiple cluster columns

- cluster_cols:

  Vector of cluster column names

- x_col:

  X-axis variable

- y_col:

  Y-axis variable

## Value

The return value of
[`grid.arrange`](https://rdrr.io/pkg/gridExtra/man/arrangeGrob.html), a
[`gtable`](https://gtable.r-lib.org/reference/gtable.html) drawn as a
side effect.

## Examples

``` r
# \donttest{
df <- iris[, 1:4]
df$km3 <- kmeans(df, 3)$cluster
df$km4 <- kmeans(df, 4)$cluster
plot_cluster_comparison(df, c("km3", "km4"), "Sepal.Length", "Sepal.Width")

# }
```
