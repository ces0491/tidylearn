# Print Method for tidy_hclust

Print Method for tidy_hclust

## Usage

``` r
# S3 method for class 'tidy_hclust'
print(x, ...)
```

## Arguments

- x:

  A tidy_hclust object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
hc <- tidy_hclust(USArrests, method = "ward.D2")
print(hc)
#> Tidy Hierarchical Clustering
#> =============================
#> 
#> Linkage method: ward.D2 
#> Distance method: euclidean 
#> Number of observations: 50 
#> Number of merges: 49 
#> 
#> Use tidy_cutree() to cut the tree and obtain cluster assignments
#> Use tidy_dendrogram() to visualize the dendrogram
# }
```
