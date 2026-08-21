# Tidy Multidimensional Scaling

Unified interface for MDS methods with tidy output

## Usage

``` r
tidy_mds(data, method = "classical", ndim = 2, distance = "euclidean", ...)
```

## Arguments

- data:

  A data frame, tibble, or distance matrix

- method:

  Character; "classical" (default), "metric", "nonmetric", "sammon", or
  "kruskal"

- ndim:

  Number of dimensions for output (default: 2)

- distance:

  Character; distance metric if data is not already a dist object
  (default: "euclidean")

- ...:

  Additional arguments passed to specific MDS functions

## Value

A list of class "tidy_mds" containing:

- config: tibble of MDS configuration (coordinates)

- stress: goodness-of-fit measure (if applicable)

- method: character string of method used

- model: original model object

## Examples

``` r
# Classical MDS
mds_result <- tidy_mds(eurodist, method = "classical")
print(mds_result)
#> Tidy MDS Analysis
#> =================
#> 
#> Method: Classical MDS 
#> Dimensions: 2 
#> Observations: 21 
#> Goodness of Fit: 75.38% 
#> 
#> Configuration (first 6 obs):
#> # A tibble: 6 × 3
#>   .obs_id     Dim1  Dim2
#>   <chr>      <dbl> <dbl>
#> 1 Athens    2290.  1799.
#> 2 Barcelona -825.   547.
#> 3 Brussels    59.2 -367.
#> 4 Calais     -82.8 -430.
#> 5 Cherbourg -352.  -291.
#> 6 Cologne    294.  -405.
```
