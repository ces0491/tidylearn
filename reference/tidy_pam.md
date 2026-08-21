# Tidy PAM (Partitioning Around Medoids)

Performs PAM clustering with tidy output

## Usage

``` r
tidy_pam(data, k, metric = "euclidean", cols = NULL)
```

## Arguments

- data:

  A data frame, tibble, or dist object

- k:

  Number of clusters

- metric:

  Distance metric (default: "euclidean"). Use "gower" for mixed data
  types.

- cols:

  Columns to include (tidy select). If NULL, uses all columns.

## Value

A list of class "tidy_pam" containing:

- clusters: tibble with observation IDs and cluster assignments

- medoids: tibble of medoid indices and values

- silhouette: average silhouette width

- model: original pam object

## Examples

``` r
# PAM with Euclidean distance
pam_result <- tidy_pam(iris, k = 3)

# PAM with Gower distance for mixed data
pam_result <- tidy_pam(mtcars, k = 3, metric = "gower")
```
