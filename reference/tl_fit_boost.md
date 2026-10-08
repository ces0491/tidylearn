# Fit a gradient boosting model

Fit a gradient boosting model

## Usage

``` r
tl_fit_boost(
  data,
  formula,
  is_classification = FALSE,
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1,
  n.minobsinnode = 10,
  cv.folds = 0,
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

- n.trees:

  Number of trees (default: 100)

- interaction.depth:

  Depth of interactions (default: 3)

- shrinkage:

  Learning rate (default: 0.1)

- n.minobsinnode:

  Minimum number of observations in terminal nodes (default: 10)

- cv.folds:

  Number of cross-validation folds (default: 0, no CV)

- ...:

  Additional arguments to pass to gbm(), including case `weights`.
  `verbose`, and for regression `distribution`, replace the defaults
  used here. An offset is refused: predict.gbm() does not add it back.

## Value

A fitted gradient boosting model

## Details

The distribution follows the response: `"gaussian"` for regression,
`"bernoulli"` for two classes and `"multinomial"` for more. gbm
describes its multinomial distribution as currently broken, kept only
for backwards compatibility, and warns to that effect on every
multiclass fit. For three or more classes, `method = "forest"` or
`method = "xgboost"` are the better supported choices.
