# Create Summary Dashboard

Generate a multi-panel summary of clustering results

## Usage

``` r
create_cluster_dashboard(
  data,
  cluster_col = "cluster",
  validation_metrics = NULL
)
```

## Arguments

- data:

  Data frame with cluster assignments

- cluster_col:

  Cluster column name

- validation_metrics:

  Optional tibble of validation metrics

## Value

Invisibly returns a list of
[`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) objects.
The combined plot grid is drawn as a side effect via
[`grid.arrange`](https://rdrr.io/pkg/gridExtra/man/arrangeGrob.html).

## Examples

``` r
# \donttest{
df <- iris[, 1:4]
df$cluster <- kmeans(df, 3)$cluster
create_cluster_dashboard(df)

# }
```
