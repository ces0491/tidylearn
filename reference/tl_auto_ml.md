# Auto ML: Automated Machine Learning Workflow

Automatically explores multiple modeling approaches including
dimensionality reduction, clustering, and various supervised methods.
Returns the best performing model, scored by cross-validation where the
time budget allows.

## Usage

``` r
tl_auto_ml(
  data,
  formula,
  task = "auto",
  use_reduction = TRUE,
  use_clustering = TRUE,
  time_budget = 300,
  cv_folds = 5,
  metric = NULL
)
```

## Arguments

- data:

  A data frame

- formula:

  Model formula (for supervised learning)

- task:

  Task type: "classification", "regression", or "auto" (default), which
  takes it from the response the formula computes, so `factor(am) ~ .`
  is a classification although `am` is numeric. A factor or character
  response is a classification and any other a regression. An explicit
  task has to agree: every candidate but logistic regression takes its
  task from the response, so a contradicting task would be scored on
  metrics the candidates cannot produce. A 0/1 numeric response is
  therefore a regression; convert it with
  [`factor()`](https://rdrr.io/r/base/factor.html) to treat its values
  as classes.

- use_reduction:

  Whether to try dimensionality reduction (default: TRUE)

- use_clustering:

  Whether to add cluster features (default: TRUE)

- time_budget:

  Time budget in seconds (default: 300). The budget is checked between
  model fits, not during them: once a model starts training it runs to
  completion, because R cannot safely interrupt C-level code
  (randomForest, xgboost, e1071). A run can therefore overshoot the
  budget by the length of the last fit it started.

  The budget gates the workflow as follows:

  - Baseline models: a tree, with linear regression for a numeric
    response or logistic regression for a two-class one. A random forest
    is added when `time_budget` is 30 or more.

  - PCA and cluster variants, when enabled: each phase starts only if
    more than `max(5, 0.1 * time_budget)` seconds remain, and fits one
    variant per baseline method while at least
    `max(2, 0.05 * time_budget)` seconds remain.

  - Advanced models (SVM and XGBoost for classification, ridge and lasso
    for regression): only when `time_budget` is 30 or more and more than
    40\\

  - Scoring: a model is cross-validated when more than 30\\ budget
    remains at the moment it is scored, and scored on its own training
    data otherwise. The leaderboard's `evaluation` column records which.

  The example below, with `time_budget = 10` on the three-class `iris`,
  fits a single tree and cross-validates it.

- cv_folds:

  Number of cross-validation folds (default: 5). Reducing this (e.g. to
  2 or 3) is an effective way to stay closer to the time budget since CV
  is typically the most expensive step.

- metric:

  Evaluation metric (default: "accuracy" for classification, "rmse" for
  regression). Classification takes "accuracy", "precision", "recall",
  "sensitivity", "specificity", "f1", "auc" or "pr_auc"; regression
  takes "rmse", "mse", "mae", "mape" or "rsq". It is checked before any
  model is fitted.

## Value

A list with class `"tidylearn_automl"` containing:

- best_model:

  The best tidylearn model object

- models:

  Named list of all successfully trained models

- leaderboard:

  Tibble ranking models by the chosen metric, with columns `model`,
  `score` and `evaluation`. The `evaluation` column records how each
  score was obtained – `"cv"` for cross-validated, `"train"` for
  training-set metrics, which are optimistic. Scores of different kinds
  are not directly comparable; a mixed leaderboard means the budget ran
  short of cross-validating every model.

- task:

  Detected or specified task type

- metric:

  Metric used for ranking

- runtime:

  Total elapsed time as a difftime object

## Details

The PCA and cluster variants are built from the formula's predictors
only, so a column the formula leaves out (`y ~ . - id`) reaches no
candidate. The cluster variants add the cluster assignment to the
formula's terms.

## Examples

``` r
# \donttest{
# Quick run with fast models only (< 30s budget skips forest/SVM/XGBoost)
result <- tl_auto_ml(iris, Species ~ .,
  time_budget = 10,
  use_reduction = FALSE,
  use_clustering = FALSE,
  cv_folds = 2)
#> Starting Auto ML with task: classification
#> Time budget: 10 seconds
#> 
#> [1/4] Training baseline models...
#>   Training: baseline_tree
#> 
#> [4/4] Training advanced models...
#> 
#> [*] Creating leaderboard...
#> 
#> Auto ML complete in 0.03 seconds
#> Best model: baseline_tree
result$leaderboard
#> # A tibble: 1 × 3
#>   model         score evaluation
#>   <chr>         <dbl> <chr>     
#> 1 baseline_tree 0.947 cv        
# }
```
