# Predict using a tidylearn model

Unified prediction interface for both supervised and unsupervised models

## Usage

``` r
# S3 method for class 'tidylearn_model'
predict(object, new_data = NULL, type = "response", ...)
```

## Arguments

- object:

  A tidylearn model object

- new_data:

  A data frame containing the new data. If NULL, uses training data.

- type:

  Type of prediction, for supervised models only: `"response"`
  (default), `"prob"` or `"class"`. Note that `"response"` is
  method-dependent – logistic regression returns probabilities, trees
  and forests return class labels – so pass `"class"` explicitly when
  you want labels. Ignored by unsupervised models, whose output is
  determined by the method.

- ...:

  Additional arguments

## Value

For supervised models, a
[tibble](https://tibble.tidyverse.org/reference/tibble.html) with a
`.pred` column; with `type = "prob"`, one column per class instead. For
unsupervised models, the method's natural output: an `.obs_id` column
plus component scores for `"pca"` and `"mds"`, or plus a `cluster`
column for the clustering methods.

Unsupervised models differ in whether they can handle new data. `"pca"`
projects it and `"kmeans"` assigns it to the nearest centre; `"pam"`,
`"clara"`, `"dbscan"`, `"mds"` and `"hclust"` have no out-of-sample
projection and error if `new_data` is supplied. For hierarchical
clustering, cut the tree with
[`tidy_cutree()`](https://tidylearn.sheetsolved.com/reference/tidy_cutree.md)
instead.

## Examples

``` r
# \donttest{
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
predict(model)
#> # A tibble: 32 × 1
#>    .pred
#>    <dbl>
#>  1  23.6
#>  2  22.6
#>  3  25.3
#>  4  21.3
#>  5  18.3
#>  6  20.5
#>  7  15.6
#>  8  22.9
#>  9  22.0
#> 10  20.0
#> # ℹ 22 more rows
predict(model, new_data = mtcars[1:5, ])
#> # A tibble: 5 × 1
#>   .pred
#>   <dbl>
#> 1  23.6
#> 2  22.6
#> 3  25.3
#> 4  21.3
#> 5  18.3
# }
```
