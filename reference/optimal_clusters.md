# Find Optimal Number of Clusters

Use multiple methods to suggest optimal k

## Usage

``` r
optimal_clusters(data, max_k = 10, methods = c("silhouette", "gap", "wss"))
```

## Arguments

- data:

  A data frame or tibble

- max_k:

  Maximum k to test (default: 10)

- methods:

  Vector of methods: "silhouette", "gap", "wss" (default: all)

## Value

A list of class `"optimal_k_results"` containing one or more of:

- wss: tibble from
  [`calc_wss`](https://tidylearn.sheetsolved.com/reference/calc_wss.md)
  (if "wss" method used)

- silhouette: tibble from
  [`tidy_silhouette_analysis`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette_analysis.md)
  (if "silhouette" method used)

- gap: a `tidy_gap` object from
  [`tidy_gap_stat`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)
  (if "gap" method used)

## Examples

``` r
# \donttest{
opt <- optimal_clusters(iris[, 1:4], max_k = 6, methods = "wss")
# }
```
