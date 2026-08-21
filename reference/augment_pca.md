# Augment Original Data with PCA Scores

Add PC scores to the original dataset

## Usage

``` r
augment_pca(pca_obj, data, n_components = NULL)
```

## Arguments

- pca_obj:

  A tidy_pca object

- data:

  Original data frame

- n_components:

  Number of PCs to add (default: all)

## Value

A tibble containing the original `data` with additional columns for each
principal component score (named `PC1`, `PC2`, etc.).

## Examples

``` r
# \donttest{
pca <- tidy_pca(USArrests)
augmented <- augment_pca(pca, USArrests, n_components = 2)
# }
```
