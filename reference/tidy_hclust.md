# Tidy Hierarchical Clustering

Performs hierarchical clustering with tidy output

## Usage

``` r
tidy_hclust(data, method = "average", distance = "euclidean", cols = NULL)
```

## Arguments

- data:

  A data frame, tibble, or dist object

- method:

  Agglomeration method: "ward.D2", "single", "complete", "average"
  (default), "mcquitty", "median", "centroid"

- distance:

  Distance metric if data is not a dist object (default: "euclidean")

- cols:

  Columns to include (tidy select). If NULL, uses all numeric columns.

## Value

A list of class "tidy_hclust" containing:

- model: hclust object

- dist: distance matrix used

- method: linkage method used

- data: original data (for plotting)

## Examples

``` r
# Basic hierarchical clustering
hc_result <- tidy_hclust(USArrests, method = "average")

# With specific distance
hc_result <- tidy_hclust(mtcars, method = "complete", distance = "manhattan")
```
