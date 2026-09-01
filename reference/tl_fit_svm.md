# Fit a support vector machine model

Fit a support vector machine model

## Usage

``` r
tl_fit_svm(
  data,
  formula,
  is_classification = FALSE,
  kernel = "radial",
  cost = 1,
  gamma = NULL,
  degree = 3,
  tune = FALSE,
  tune_folds = 5,
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

- kernel:

  Kernel function ("linear", "polynomial", "radial", "sigmoid")

- cost:

  Cost parameter (default: 1)

- gamma:

  Gamma parameter for kernels. Left to
  [`e1071::svm()`](https://rdrr.io/pkg/e1071/man/svm.html) when `NULL`,
  which uses 1 divided by the number of columns in the design matrix.

- degree:

  Degree for polynomial kernel (default: 3)

- tune:

  Logical indicating whether to tune hyperparameters (default: FALSE)

- tune_folds:

  Number of folds for cross-validation during tuning (default: 5)

- ...:

  Additional arguments to pass to svm()

## Value

A fitted SVM model
