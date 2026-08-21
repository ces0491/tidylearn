# Plot Dendrogram

Create dendrogram visualization

## Usage

``` r
tidy_dendrogram(hclust_obj, k = NULL, hang = 0.01, cex = 0.7)
```

## Arguments

- hclust_obj:

  A tidy_hclust object or hclust object

- k:

  Optional; number of clusters to highlight with rectangles

- hang:

  Fraction of plot height to hang labels (default: 0.01)

- cex:

  Label size (default: 0.7)

## Value

The [`hclust`](https://rdrr.io/r/stats/hclust.html) object, returned
invisibly. The dendrogram is plotted as a side effect.

## Examples

``` r
# \donttest{
hc <- tidy_hclust(USArrests, method = "ward.D2")
tidy_dendrogram(hc, k = 3)

# }
```
