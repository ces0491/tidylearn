# Calculate Within-Cluster Sum of Squares for Different k

Used for elbow method to determine optimal k

## Usage

``` r
calc_wss(data, max_k = 10, nstart = 25)
```

## Arguments

- data:

  A data frame or tibble

- max_k:

  Maximum number of clusters to test (default: 10)

- nstart:

  Number of random starts for each k (default: 25)

## Value

A tibble with columns `k` (number of clusters) and `tot_withinss` (total
within-cluster sum of squares).

## Examples

``` r
# \donttest{
wss <- calc_wss(iris[, 1:4], max_k = 6)
plot(wss$k, wss$tot_withinss, type = "b")

# }
```
