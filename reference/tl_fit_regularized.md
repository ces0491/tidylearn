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
  ...,
  weights = NULL,
  foldid = NULL,
  subset = NULL,
  offset = NULL
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

  Regularization parameter: a single penalty to fit at, or `NULL` (the
  default) or a sequence of penalties, from which cross-validation
  chooses one

- cv_folds:

  Number of folds for cross-validation (default: 5)

- ...:

  Additional arguments to pass to glmnet() or cv.glmnet(). A name
  neither function takes is an error, as are `x`, `y`, `family` and
  `nfolds`, which tidylearn sets itself, and `relax = TRUE` and `gamma`:
  predictions and coefficients come from the unrelaxed fit.

- weights:

  Optional case weights, one per row of `data`

- foldid:

  Optional fold for each row of `data`, for the cross-validation that
  chooses lambda

- subset:

  Optional rows of `data` to fit on

- offset:

  Not supported: glmnet would need the offset again at every prediction.
  An error if supplied.

## Value

A fitted regularized regression model
