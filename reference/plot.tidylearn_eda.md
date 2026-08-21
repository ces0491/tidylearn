# Plot EDA results

Plot EDA results

## Usage

``` r
# S3 method for class 'tidylearn_eda'
plot(x, ...)
```

## Arguments

- x:

  A tidylearn_eda object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly. Called for its side effect of
plotting a PCA scatter plot coloured by cluster.

## Examples

``` r
# \donttest{
eda <- tl_explore(iris, response = "Species")
#> Running Exploratory Data Analysis...
#> [1/4] PCA analysis...
#> [2/4] Finding optimal clusters...
#> [3/4] Clustering analysis...
#> [4/4] Distance analysis...
#> EDA complete!
plot(eda)

# }
```
