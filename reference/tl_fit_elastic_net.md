# Fit an Elastic Net regression model

Fit an Elastic Net regression model

## Usage

``` r
tl_fit_elastic_net(
  data,
  formula,
  is_classification = FALSE,
  alpha = 0.5,
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

  Mixing parameter (default: 0.5 for Elastic Net)

- lambda:

  Regularization parameter: a single penalty, or NULL or a sequence of
  penalties for cross-validation to choose from

- cv_folds:

  Number of folds for cross-validation (default: 5)

- ...:

  Additional arguments to pass to glmnet() or cv.glmnet()

## Value

A fitted Elastic Net regression model
