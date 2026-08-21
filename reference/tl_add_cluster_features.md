# Cluster-Based Features

Add cluster assignments as features for supervised learning. This
semi-supervised approach can capture non-linear patterns.

## Usage

``` r
tl_add_cluster_features(data, response = NULL, method = "kmeans", ...)
```

## Arguments

- data:

  A data frame

- response:

  Response variable name (will be excluded from clustering)

- method:

  Clustering method: "kmeans", "pam", "hclust", "dbscan"

- ...:

  Additional arguments for clustering

## Value

The original data frame with an additional factor column named
`cluster_<method>` containing cluster assignments. The fitted cluster
model is stored as an attribute `"cluster_model"`.

## Examples

``` r
# \donttest{
# Add cluster features before supervised learning
data_with_clusters <- tl_add_cluster_features(iris, response = "Species",
                                                method = "kmeans", k = 3)
model <- tl_model(data_with_clusters, Species ~ ., method = "forest")
# }
```
