# Tidy DBSCAN Clustering

Performs density-based clustering with tidy output

## Usage

``` r
tidy_dbscan(data, eps, minPts = 5, cols = NULL, distance = "euclidean")
```

## Arguments

- data:

  A data frame, tibble, numeric matrix, or dist object

- eps:

  Neighborhood radius (epsilon)

- minPts:

  Minimum number of points to form a dense region (default: 5)

- cols:

  Columns to include (tidy select). If NULL, uses all numeric columns,
  or every column for `distance = "gower"`.

- distance:

  Distance metric if data is not a dist object (default: "euclidean"):
  any method [`dist`](https://rdrr.io/r/stats/dist.html) accepts, or
  "gower" for mixed data types

## Value

A list of class "tidy_dbscan" containing:

- clusters: tibble with observation IDs, cluster assignments (0 =
  noise), and the logical flags `is_noise` and `is_core`

- summary: tibble with each cluster's size and number of core points

- n_clusters: number of clusters (excluding noise)

- n_noise: number of noise points

- eps, minPts: the parameters used

- model: original dbscan object

## Examples

``` r
# Basic DBSCAN
db_result <- tidy_dbscan(iris, eps = 0.5, minPts = 5)

# With suggested eps from k-NN distance plot
eps_suggestion <- suggest_eps(iris, minPts = 5)
db_result <- tidy_dbscan(iris, eps = eps_suggestion$eps, minPts = 5)
```
