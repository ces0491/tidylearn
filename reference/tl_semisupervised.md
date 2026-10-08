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
  ...,
  cluster_args = list()
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula. The response must be categorical: a factor, character
  or logical column, or an expression that computes a factor or
  character vector from a column, such as `factor(am)` or
  `factor(mpg > 20)`. For an expression, each row's propagated label is
  written to that column as its value in a labelled row of the same
  class. An expression that then computes other labels, such as
  `cut(mpg, 2)`, whose breaks follow the column's range, is refused.

- labeled_indices:

  Indices of labeled observations

- cluster_method:

  Clustering method for label propagation: `"kmeans"` (default),
  `"pam"`, `"clara"` or `"hclust"`, whose tree is cut at k

- supervised_method:

  Supervised learning method for the final model (default: `"tree"`,
  which handles any number of classes). `"logistic"` is binary-only and
  errors on a response with more than two levels.

- ...:

  Additional arguments for the supervised model

- cluster_args:

  A named list of arguments for the clustering step, such as
  `list(nstart = 5)` for k-means. k is set from the labelled classes and
  cannot be given here.

## Value

A tidylearn model object with additional class
`"tidylearn_semisupervised"`, trained on pseudo-labeled data. The model
includes a `semisupervised_info` element with `labeled_indices`,
`cluster_model`, `label_mapping`, and `n_unlabelled_dropped`, the number
of rows left out for having no label to train on.

## Details

Labels are propagated by majority vote within each cluster, so the
response must be categorical. The rows are clustered on the predictors
the formula names, into as many clusters as the labelled rows hold
classes. A labelled row whose label is missing takes no part in the
vote. Rows in a cluster where no labelled observation carries a label
have no label to take, and labelled rows whose own label is missing have
none either; both are left out of training, with a warning giving the
counts. The pseudo-labels keep the response's level order, so the second
level stays the positive class.

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
