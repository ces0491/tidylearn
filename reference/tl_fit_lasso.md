# Fit a Lasso regression model

Fit a Lasso regression model

## Usage

``` r
tl_fit_lasso(
  data,
  formula,
  is_classification = FALSE,
  alpha = 1,
  lambda = NULL,
  cv_folds = 5,
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

- alpha:

  Mixing parameter (0 for Ridge, 1 for Lasso, between 0-1 for Elastic
  Net)

- lambda:

  Regularization parameter: a single penalty, or NULL or a sequence of
  penalties for cross-validation to choose from

- cv_folds:

  Number of folds for cross-validation (default: 5)

- ...:

  Additional arguments to pass to glmnet() or cv.glmnet()

## Value

A fitted Lasso regression model
