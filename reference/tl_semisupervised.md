# Semi-Supervised Learning via Clustering

Train a supervised model with limited labels by first clustering the
data and propagating labels within clusters.

## Usage

``` r
tl_semisupervised(
  data,
  formula,
  labeled_indices,
  cluster_method = "kmeans",
  supervised_method = "tree",
  ...
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula. The response must be a factor, character or logical
  column.

- labeled_indices:

  Indices of labeled observations

- cluster_method:

  Clustering method for label propagation

- supervised_method:

  Supervised learning method for the final model (default: `"tree"`,
  which handles any number of classes). `"logistic"` is binary-only and
  errors on a response with more than two levels.

- ...:

  Additional arguments

## Value

A tidylearn model object with additional class
`"tidylearn_semisupervised"`, trained on pseudo-labeled data. The model
includes a `semisupervised_info` element with `labeled_indices`,
`cluster_model`, `label_mapping`, and `n_unlabelled_dropped`, the number
of rows left out because their cluster had no labelled observation.

## Details

Labels are propagated by majority vote within each cluster, so the
response must be categorical. Rows in a cluster that holds no labelled
observation have no label to take; they are left out of training, with a
warning giving the count.

## Examples

``` r
# \donttest{
# Use only 10% of labels
labeled_idx <- sample(nrow(iris), size = 15)
model <- tl_semisupervised(iris, Species ~ ., labeled_indices = labeled_idx,
  cluster_method = "kmeans",
  supervised_method = "tree"
)
# }
```
