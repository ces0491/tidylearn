# Automated Machine Learning with tidylearn

``` r

library(tidylearn)
library(dplyr)
```

## Introduction

[`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
searches across methods rather than within one. It trains several
models, engineers a couple of feature sets, cross-validates what the
budget allows, and ranks the results.

It orchestrates the wrapped packages rather than implementing anything
new. Every model on the leaderboard is an ordinary tidylearn model, so
[`predict()`](https://rdrr.io/r/stats/predict.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) and `$fit` all
work on it.

## A First Run

``` r

result <- tl_auto_ml(
  iris, Species ~ .,
  time_budget = 30,
  cv_folds = 3
)
```

``` r

result$leaderboard
#> # A tibble: 8 × 3
#>   model            score evaluation
#>   <chr>            <dbl> <chr>     
#> 1 clustered_tree   0.953 cv        
#> 2 baseline_tree    0.947 cv        
#> 3 baseline_forest  0.947 cv        
#> 4 clustered_forest 0.947 cv        
#> 5 advanced_svm     0.947 cv        
#> 6 advanced_xgboost 0.947 cv        
#> 7 pca_tree         0.9   cv        
#> 8 pca_forest       0.873 cv
```

Three columns: which model, what it scored, and how the score was
obtained.

``` r

result$best_model
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: tree 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 150
```

``` r

result$task
#> [1] "classification"
result$metric
#> [1] "accuracy"
round(as.numeric(result$runtime, units = "secs"), 1)
#> [1] 1.1
```

Regression is the same call with a numeric response — `task = "auto"`
reads it off the data:

``` r

result_reg <- tl_auto_ml(
  mtcars, mpg ~ .,
  time_budget = 30,
  cv_folds = 3
)

result_reg$task
#> [1] "regression"
```

``` r

result_reg$leaderboard
#> # A tibble: 11 × 3
#>    model            score evaluation
#>    <chr>            <dbl> <chr>     
#>  1 advanced_ridge    2.66 cv        
#>  2 clustered_forest  2.67 cv        
#>  3 baseline_forest   2.70 cv        
#>  4 pca_linear        2.97 cv        
#>  5 advanced_lasso    2.97 cv        
#>  6 baseline_linear   3.32 cv        
#>  7 baseline_tree     4.11 cv        
#>  8 pca_forest        4.16 cv        
#>  9 clustered_linear  4.34 cv        
#> 10 clustered_tree    4.48 cv        
#> 11 pca_tree          4.65 cv
```

## How the Search Works

Four phases run in order, each gated on the remaining budget and on the
`use_reduction` and `use_clustering` toggles.

| Phase | What it does | Classification | Regression |
|----|----|----|----|
| 1\. Baselines | Standard models on the raw features | tree, logistic\*, forest | tree, linear, forest |
| 2\. PCA variants | PCA preprocessing, then the baselines | pca_tree, pca_logistic\*, pca_forest | pca_tree, pca_linear, pca_forest |
| 3\. Cluster variants | Cluster assignment added as a feature | clustered_tree, clustered_logistic\*, clustered_forest | clustered_tree, clustered_linear, clustered_forest |
| 4\. Advanced | Heavier methods | svm, xgboost | ridge, lasso |

\* Logistic regression is binary only, so it appears only when the
response has exactly two classes. On a multiclass problem such as `iris`
the baselines are tree and forest.

You can see the phases in what actually got trained:

``` r

names(result$models)
#> [1] "baseline_tree"    "baseline_forest"  "pca_tree"         "pca_forest"      
#> [5] "clustered_tree"   "clustered_forest" "advanced_svm"     "advanced_xgboost"
```

``` r

# The same search on a two-class problem picks up the logistic variants
iris_binary <- iris %>%
  filter(Species != "setosa") %>%
  mutate(Species = droplevels(Species))

binary_result <- tl_auto_ml(iris_binary, Species ~ ., time_budget = 30,
                            cv_folds = 3)
names(binary_result$models)
#>  [1] "baseline_tree"      "baseline_logistic"  "baseline_forest"   
#>  [4] "pca_tree"           "pca_logistic"       "pca_forest"        
#>  [7] "clustered_tree"     "clustered_logistic" "clustered_forest"  
#> [10] "advanced_svm"       "advanced_xgboost"
```

That call emits a run of `glm.fit: algorithm did not converge` warnings,
which this chunk hides only because there are a dozen of them. They are
worth understanding rather than ignoring: *versicolor* and *virginica*
overlap only slightly, and a three-fold split of 100 rows will sometimes
produce a fold where the two are perfectly separable. Logistic
regression has no finite maximum likelihood estimate on separable data,
so the coefficients diverge and
[`glm()`](https://rdrr.io/r/stats/glm.html) says so.

The fit on the full data is fine:

``` r

tl_model(iris_binary, Species ~ ., method = "logistic")$spec$method
#> [1] "logistic"
```

It is the folds, not the dataset. Seeing this on your own data means the
classes are close to separable at that sample size — a reason to prefer
a regularised method, which is what `ridge` and `lasso` are doing in
phase 4 of the regression search.

### The evaluation column

Each model is fitted on the full training data, then cross-validated if
the budget allows. When it does not, the leaderboard falls back to
training-set metrics.

``` r

table(result$leaderboard$evaluation)
#> 
#> cv 
#>  8
```

Training metrics are optimistically biased, so a leaderboard mixing
`"cv"` and `"train"` is not ranking like with like. If you see both,
raise the budget or lower `cv_folds`.

## The Time Budget

`time_budget` is in seconds and is checked **between** model fits, not
during them. Once a model starts it runs to completion, because the
wrapped packages execute C-level code that R cannot safely interrupt.
Wall-clock time can therefore overshoot by the duration of whichever
model started last.

The budget controls how much gets tried. Rather than quote a table of
predictions, here is the actual relationship on this data and this
machine:

``` r

budgets <- c(2, 5, 10, 30)

sweep <- lapply(budgets, function(b) {
  t0 <- Sys.time()
  r <- tl_auto_ml(iris, Species ~ ., time_budget = b, cv_folds = 3)
  data.frame(
    budget = b,
    elapsed = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    models = nrow(r$leaderboard),
    cv_scored = sum(r$leaderboard$evaluation == "cv"),
    best = r$leaderboard$model[1]
  )
})

do.call(rbind, sweep)
#>   budget elapsed models cv_scored             best
#> 1      2     0.0      1         1    baseline_tree
#> 2      5     0.0      1         1    baseline_tree
#> 3     10     0.2      3         3    baseline_tree
#> 4     30     0.7      8         8 clustered_forest
```

iris is 150 rows, so everything here is quick and the budget is barely
touched. On data where a single forest fit takes ten seconds the same
numbers look very different — which is the point of running the sweep on
your own data rather than trusting a table.

Two things are worth knowing regardless of size:

- The forest baseline and every advanced model are only attempted when
  `time_budget >= 30`.
- Cross-validation is the expensive step. `tl_cv(folds = 5)` fits five
  models where a plain fit costs one, so `cv_folds` is the most
  effective lever.

## Controlling the Search

``` r

# Baselines only -- no PCA, no cluster features
baseline_only <- tl_auto_ml(
  iris, Species ~ .,
  use_reduction = FALSE,
  use_clustering = FALSE,
  time_budget = 30,
  cv_folds = 3
)

names(baseline_only$models)
#> [1] "baseline_tree"    "baseline_forest"  "advanced_svm"     "advanced_xgboost"
```

``` r

# Keep PCA, drop clustering
no_clustering <- tl_auto_ml(
  iris, Species ~ .,
  use_clustering = FALSE,
  time_budget = 30,
  cv_folds = 3
)

names(no_clustering$models)
#> [1] "baseline_tree"    "baseline_forest"  "pca_tree"         "pca_forest"      
#> [5] "advanced_svm"     "advanced_xgboost"
```

Naming a metric changes what the leaderboard optimises. Classification
accepts `accuracy`, `precision`, `recall`, `sensitivity`, `specificity`,
`f1`, `auc` and `pr_auc`; regression accepts `rmse`, `mse`, `mae`,
`mape` and `rsq`.

``` r

by_f1 <- tl_auto_ml(iris_binary, Species ~ ., metric = "f1",
                    time_budget = 30, cv_folds = 3)

by_f1$metric
#> [1] "f1"
by_f1$leaderboard
#> # A tibble: 11 × 3
#>    model              score evaluation
#>    <chr>              <dbl> <chr>     
#>  1 advanced_svm       0.947 cv        
#>  2 clustered_logistic 0.932 cv        
#>  3 baseline_logistic  0.928 cv        
#>  4 clustered_tree     0.928 cv        
#>  5 clustered_forest   0.923 cv        
#>  6 advanced_xgboost   0.921 cv        
#>  7 baseline_forest    0.905 cv        
#>  8 baseline_tree      0.893 cv        
#>  9 pca_forest         0.888 cv        
#> 10 pca_logistic       0.845 cv        
#> 11 pca_tree           0.764 cv
```

## Using the Result

``` r

split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)

automl <- tl_auto_ml(split$train, Species ~ ., time_budget = 30, cv_folds = 3)

test_preds <- predict(automl$best_model, new_data = split$test)
mean(test_preds$.pred == split$test$Species)
#> [1] 0.9333333
```

Any model on the leaderboard can be pulled out by name:

``` r

available <- names(automl$models)
available
#> [1] "baseline_tree"    "baseline_forest"  "pca_tree"         "pca_forest"      
#> [5] "clustered_tree"   "clustered_forest" "advanced_svm"     "advanced_xgboost"
```

``` r

scores <- vapply(available, function(nm) {
  preds <- predict(automl$models[[nm]], new_data = split$test)
  mean(preds$.pred == split$test$Species)
}, numeric(1))

data.frame(model = available, test_accuracy = round(scores, 3),
           row.names = NULL) %>%
  arrange(desc(test_accuracy))
#>              model test_accuracy
#> 1    baseline_tree         0.933
#> 2  baseline_forest         0.933
#> 3   clustered_tree         0.933
#> 4     advanced_svm         0.933
#> 5 clustered_forest         0.911
#> 6 advanced_xgboost         0.911
#> 7         pca_tree         0.889
#> 8       pca_forest         0.844
```

Comparing the leaderboard against held-out accuracy is worth doing: the
model that won the search is not always the one that generalises, and
with a dataset this small the difference between the top few is noise.

## AutoML Against a Single Choice

``` r

manual <- tl_model(split$train, Species ~ ., method = "forest")
manual_acc <- mean(
  predict(manual, new_data = split$test)$.pred == split$test$Species
)

automl_acc <- mean(test_preds$.pred == split$test$Species)

data.frame(
  approach = c("forest, chosen by hand", "AutoML best"),
  test_accuracy = round(c(manual_acc, automl_acc), 3)
)
#>                 approach test_accuracy
#> 1 forest, chosen by hand         0.933
#> 2            AutoML best         0.933
```

On iris a random forest is already the right answer, so the search does
not improve on it. That is the expected result on an easy,
well-understood dataset — AutoML earns its cost when you do not already
know which method suits the data.

## Preprocessing First

AutoML does not impute. Handle missing values before the call, and apply
the same transformation to any data you later predict on:

``` r

processed <- tl_prepare_data(
  split$train, Species ~ .,
  scale_method = "standardize",
  remove_correlated = TRUE
)

automl_processed <- tl_auto_ml(processed$data, Species ~ .,
                               time_budget = 30, cv_folds = 3)

automl_processed$leaderboard$model[1]
#> [1] "baseline_tree"
```

[`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
handles that bookkeeping properly — see
[`vignette("tuning-and-pipelines")`](https://tidylearn.sheetsolved.com/articles/tuning-and-pipelines.md).

## When a Model Is Missing or Scores NA

A model that fails to fit, or errors during evaluation, is dropped with
a message explaining why and never reaches the leaderboard. Run without
`message = FALSE` to see which and why.

A model that evaluates but produces no value for the chosen metric
appears with an `NA` score. That usually means the metric is not one
[`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
recognises for the task. If every score is `NA`,
[`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
warns and returns the first model trained rather than pretending to have
ranked them.

``` r

sum(is.na(result$leaderboard$score))
#> [1] 0
```

## AutoML or Tuning?

They search different axes and compose in one direction:

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  searches **across methods** at default hyperparameters.
- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  search **within one method**.

Run AutoML first to find out which family suits the data, then tune the
winner:

``` r

winner <- automl$best_model$spec$method
winner
#> [1] "forest"
```

``` r

tuned <- tl_tune_grid(
  split$train, Species ~ .,
  method = winner,
  param_grid = tl_default_param_grid(winner, size = "small"),
  folds = 3,
  verbose = FALSE
)

attr(tuned, "tuning_results")$best_params
#> $mtry
#> [1] 2
#> 
#> $ntree
#> [1] 100
```

``` r

mean(predict(tuned, new_data = split$test)$.pred == split$test$Species)
#> [1] 0.9333333
```

## Guidance

1.  **Start at `time_budget = 10`** to confirm the call is well formed,
    then raise it. Nothing about a 10-second run tells you which model
    is best.
2.  **Lower `cv_folds` before lowering `time_budget`.** Going from 5
    folds to 2 removes 60% of the evaluation cost and still gives
    out-of-sample estimates. A 30-second budget at `cv_folds = 2` is
    more informative than the same budget at the default 5, which will
    skip CV entirely.
3.  **Preprocess first.** AutoML does not impute or scale.
4.  **Score on held-out data.** The leaderboard ranks; it does not
    report final performance.
5.  **Read past the first row.** When the top few scores are within
    noise of each other, pick on interpretability or fit time instead.
6.  **Check the `evaluation` column** before comparing scores.

## Where to Go Next

- **Tuning and pipelines**
  ([`vignette("tuning-and-pipelines")`](https://tidylearn.sheetsolved.com/articles/tuning-and-pipelines.md))
  — search within a method, then freeze the recipe.
- **Diagnostics**
  ([`vignette("diagnostics")`](https://tidylearn.sheetsolved.com/articles/diagnostics.md))
  — what to check about the model AutoML handed you.
- **Integration workflows**
  ([`vignette("integration-workflows")`](https://tidylearn.sheetsolved.com/articles/integration-workflows.md))
  — the PCA and clustering steps AutoML applies, driven by hand.
