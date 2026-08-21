# Tidy CLARA (Clustering Large Applications)

Performs CLARA clustering (scalable version of PAM)

## Usage

``` r
tidy_clara(data, k, metric = "euclidean", samples = 50, sampsize = NULL)
```

## Arguments

- data:

  A data frame or tibble

- k:

  Number of clusters

- metric:

  Distance metric (default: "euclidean")

- samples:

  Number of samples to draw (default: 50)

- sampsize:

  Sample size (default: min(n, 40 + 2\*k))

## Value

A list of class `"tidy_clara"` containing:

- clusters: tibble with observation IDs and cluster assignments

- medoids: tibble of medoid values

- silhouette_avg: average silhouette width

- model: original [`clara`](https://rdrr.io/pkg/cluster/man/clara.html)
  object

## Examples

``` r
# \donttest{
# CLARA for large datasets
large_data <- iris[rep(1:nrow(iris), 10), 1:4]
clara_result <- tidy_clara(large_data, k = 3, samples = 50)
print(clara_result)
#> $clusters
#> # A tibble: 1,500 × 2
#>    .obs_id cluster
#>    <chr>     <int>
#>  1 1             1
#>  2 2             1
#>  3 3             1
#>  4 4             1
#>  5 5             1
#>  6 6             1
#>  7 7             1
#>  8 8             1
#>  9 9             1
#> 10 10            1
#> # ℹ 1,490 more rows
#> 
#> $medoids
#> # A tibble: 3 × 5
#>   cluster Sepal.Length Sepal.Width Petal.Length Petal.Width
#>     <int>        <dbl>       <dbl>        <dbl>       <dbl>
#> 1       1          5           3.4          1.5         0.2
#> 2       2          6           2.9          4.5         1.5
#> 3       3          6.8         3            5.5         2.1
#> 
#> $silhouette_avg
#> [1] 0.5615558
#> 
#> $model
#> Call:     cluster::clara(x = data_numeric, k = k, metric = metric, samples = samples, sampsize = sampsize) 
#> Medoids:
#>       Sepal.Length Sepal.Width Petal.Length Petal.Width
#> 8.3            5.0         3.4          1.5         0.2
#> 79.3           6.0         2.9          4.5         1.5
#> 113.8          6.8         3.0          5.5         2.1
#> Objective function:   0.6542077
#> Clustering vector:    Named int [1:1500] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
#>  - attr(*, "names")= chr [1:1500] "1" "2" "3" "4" "5" "6" "7" ...
#> Cluster sizes:            500 620 380 
#> Best sample:
#>  [1] 18    35    122   46.1  74.1  107.1 120.1 2.2   67.2  1.3   8.3   32.3 
#> [13] 38.3  61.3  79.3  109.3 110.3 134.3 150.3 15.4  82.4  107.4 109.4 137.4
#> [25] 139.4 21.5  80.5  134.5 42.6  74.6  96.6  62.7  80.7  90.7  97.7  129.7
#> [37] 140.7 47.8  69.8  108.8 113.8 140.8 67.9  71.9  144.9 150.9
#> 
#> Available components:
#>  [1] "sample"     "medoids"    "i.med"      "clustering" "objective" 
#>  [6] "clusinfo"   "diss"       "call"       "silinfo"    "data"      
#> 
#> attr(,"class")
#> [1] "tidy_clara" "list"      
# }
```
