# Tidy PAM (Partitioning Around Medoids)

Performs PAM clustering with tidy output

## Usage

``` r
tidy_pam(data, k, metric = "euclidean", cols = NULL, ...)
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

- ...:

  Further arguments passed to
  [`pam`](https://rdrr.io/pkg/cluster/man/pam.html), such as `nstart`,
  `variant` or starting `medoids`. `cluster.only = TRUE` and `diss` are
  refused, under any abbreviation R would accept: the first leaves no
  fit to build the result from, and tidy_pam() sets the second itself
  from `data`.

## Value

A list of class "tidy_pam" containing:

- clusters: tibble with observation IDs and cluster assignments

- medoids: tibble with one row per cluster: the medoid's row position in
  the data (`medoid_index`, an integer) and, unless `data` was a dist
  object, its values

- silhouette_avg: average silhouette width

- silhouette_data: the silhouette information `pam()` returns (its
  `silinfo`)

- model: original pam object

## Examples

``` r
# PAM with Euclidean distance
pam_result <- tidy_pam(iris, k = 3)

# PAM with Gower distance for mixed data
pam_result <- tidy_pam(mtcars, k = 3, metric = "gower")
```
