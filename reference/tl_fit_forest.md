# Fit a random forest model

Fit a random forest model

## Usage

``` r
tl_fit_forest(
  data,
  formula,
  is_classification = FALSE,
  ntree = 500,
  mtry = NULL,
  importance = TRUE,
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

- ntree:

  Number of trees to grow (default: 500)

- mtry:

  Number of variables randomly sampled at each split

- importance:

  Whether to compute variable importance (default: TRUE)

- ...:

  Additional arguments to pass to randomForest()

## Value

A fitted random forest model
