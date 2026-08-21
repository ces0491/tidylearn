# Tidy K-Means Clustering

Performs k-means clustering with tidy output

## Usage

``` r
tidy_kmeans(
  data,
  k,
  cols = NULL,
  nstart = 25,
  iter_max = 100,
  algorithm = "Hartigan-Wong"
)
```

## Arguments

- data:

  A data frame or tibble

- k:

  Number of clusters

- cols:

  Columns to include (tidy select). If NULL, uses all numeric columns.

- nstart:

  Number of random starts (default: 25)

- iter_max:

  Maximum iterations (default: 100)

- algorithm:

  K-means algorithm: "Hartigan-Wong" (default), "Lloyd", "Forgy",
  "MacQueen"

## Value

A list of class "tidy_kmeans" containing:

- clusters: tibble with observation IDs and cluster assignments

- centers: tibble of cluster centers

- metrics: tibble with clustering quality metrics

- model: original kmeans object

## Examples

``` r
# Basic k-means
km_result <- tidy_kmeans(iris, k = 3)
```
