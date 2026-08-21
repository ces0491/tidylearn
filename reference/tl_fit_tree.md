# Fit a decision tree model

Fit a decision tree model

## Usage

``` r
tl_fit_tree(
  data,
  formula,
  is_classification = FALSE,
  cp = 0.01,
  minsplit = 20,
  maxdepth = 30,
  ...
)
```

## Arguments

- data:

  A data frame containing the training data

- formula:

  A formula specifying the model

- is_classification:

  Logical indicating if this is a classification problem

- cp:

  Complexity parameter (default: 0.01)

- minsplit:

  Minimum number of observations in a node for a split

- maxdepth:

  Maximum depth of the tree

- ...:

  Additional arguments to pass to rpart()

## Value

A fitted decision tree model
