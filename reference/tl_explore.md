# Exploratory Data Analysis Workflow

Comprehensive EDA combining unsupervised learning techniques to
understand data structure before modeling

## Usage

``` r
tl_explore(data, response = NULL, max_components = 5, k_range = 2:6)
```

## Arguments

- data:

  A data frame

- response:

  Optional response variable for colored visualizations

- max_components:

  Maximum PCA components to compute (default: 5)

- k_range:

  Range of k values for clustering (default: 2:6)

## Value

A list with class `"tidylearn_eda"` containing:

- data:

  The original data frame.

- response:

  The response variable name, or `NULL`.

- pca:

  The fitted PCA model.

- optimal_k:

  List with optimal cluster count results.

- kmeans:

  The fitted k-means model.

- hclust:

  The fitted hierarchical clustering model.

- summary:

  List with `n_obs`, `n_vars`, `n_components`, and `best_k`.

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
