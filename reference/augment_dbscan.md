# Augment Data with DBSCAN Cluster Assignments

Augment Data with DBSCAN Cluster Assignments

## Usage

``` r
augment_dbscan(dbscan_obj, data)
```

## Arguments

- dbscan_obj:

  A tidy_dbscan object

- data:

  Original data frame

## Value

A tibble containing the original `data` with additional columns
`cluster` (factor), `is_noise` (logical), and `is_core` (logical).

## Examples

``` r
# \donttest{
db <- tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)
augmented <- augment_dbscan(db, iris)
# }
```
