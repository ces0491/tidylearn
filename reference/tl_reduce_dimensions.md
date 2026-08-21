# Integration Functions: Combining Supervised and Unsupervised Learning

These functions demonstrate the power of tidylearn's unified approach by
seamlessly integrating supervised and unsupervised learning techniques.
Feature Engineering via Dimensionality Reduction

## Usage

``` r
tl_reduce_dimensions(
  data,
  response = NULL,
  method = "pca",
  n_components = NULL,
  ...
)
```

## Arguments

- data:

  A data frame

- response:

  Response variable name (will be preserved)

- method:

  Dimensionality reduction method: "pca", "mds"

- n_components:

  Number of components to retain

- ...:

  Additional arguments for the dimensionality reduction method

## Value

A list with components:

- data:

  The transformed data frame with reduced-dimension columns and the
  response variable (if provided).

- reduction_model:

  The fitted tidylearn dimensionality reduction model.

- original_data:

  The original input data frame.

- response:

  The response variable name, or `NULL`.

## Details

Use PCA, MDS, or other dimensionality reduction as a preprocessing step
for supervised learning. This can improve model performance and
interpretability.

## Examples

``` r
# \donttest{
# Reduce dimensions before classification
reduced <- tl_reduce_dimensions(
  iris, response = "Species",
  method = "pca", n_components = 3
)
model <- tl_model(reduced$data, Species ~ ., method = "tree")
# }
```
