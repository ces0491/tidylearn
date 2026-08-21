# Fit a regularized regression model

Fits Ridge, Lasso, or Elastic Net regularization.

## Usage

``` r
tl_fit_regularized(
  data,
  formula,
  is_classification = FALSE,
  alpha = 0,
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

  Regularization parameter (if NULL, uses cross-validation to select)

- cv_folds:

  Number of folds for cross-validation (default: 5)

- ...:

  Additional arguments to pass to glmnet()

## Value

A fitted regularized regression model
