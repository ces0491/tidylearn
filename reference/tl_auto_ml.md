# Auto ML: Automated Machine Learning Workflow

Automatically explores multiple modeling approaches including
dimensionality reduction, clustering, and various supervised methods.
Returns the best performing model based on cross-validation.

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

  Task type: "classification", "regression", or "auto" (default)

- use_reduction:

  Whether to try dimensionality reduction (default: TRUE)

- use_clustering:

  Whether to add cluster features (default: TRUE)

- time_budget:

  Time budget in seconds (default: 300). Controls which models are
  attempted and whether cross-validation is used for evaluation. The
  budget is checked **between** model fits, not during them – once a
  model starts training it runs to completion because R cannot safely
  interrupt C-level code (e.g. randomForest, xgboost, e1071).

  How the budget shapes the workflow:

  - **Under 30s**: Only fast models are attempted (tree,
    logistic/linear). Cross-validation is skipped; models are ranked on
    training-set metrics only. Expect 2 models in the leaderboard. Use
    this for quick sanity checks or interactive exploration.

  - **30–120s**: All baseline models are attempted including random
    forest. Cross-validation runs when enough time remains after each
    model fit; otherwise training metrics are used. Advanced models
    (SVM, XGBoost / ridge, lasso) are attempted if 40\\ remains after
    baselines. Dimensionality reduction and clustering pipelines run if
    enabled and 10\\

  - **120s+ (recommended)**: The full pipeline runs – all baselines,
    advanced models, PCA-augmented variants, and cluster-augmented
    variants, each with cross-validation. Expect 9–11 models in the
    leaderboard.

  Because individual model fits (especially forest, SVM, XGBoost
  with CV) can take 5–30s each depending on data size, the actual
  wall-clock time may modestly exceed the budget by the duration of the
  last model that was started before the budget expired.

- cv_folds:

  Number of cross-validation folds (default: 5). Reducing this (e.g. to
  2 or 3) is an effective way to stay closer to the time budget since CV
  is typically the most expensive step.

- metric:

  Evaluation metric (default: auto-selected based on task). For
  classification: "accuracy"; for regression: "rmse".

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
#> Auto ML complete in 0.02 seconds
#> Best model: baseline_tree
result$leaderboard
#> # A tibble: 1 × 3
#>   model         score evaluation
#>   <chr>         <dbl> <chr>     
#> 1 baseline_tree 0.933 cv        
# }
```
