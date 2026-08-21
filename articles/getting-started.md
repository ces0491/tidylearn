# Getting Started with tidylearn

## Introduction

`tidylearn` provides a **unified tidyverse-compatible interface** to R’s
machine learning ecosystem. It wraps proven packages like glmnet,
randomForest, xgboost, e1071, cluster, and dbscan - you get the
reliability of established implementations with the convenience of a
consistent, tidy API.

**What tidylearn does:**

- Provides one consistent interface
  ([`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md))
  to 20 ML algorithms
- Returns tidy tibbles instead of varied output formats
- Offers unified ggplot2-based visualization across all methods
- Enables pipe-friendly workflows

**What tidylearn is NOT:**

- A reimplementation of ML algorithms (uses established packages under
  the hood)
- A replacement for the underlying packages (you can access the raw
  model via `model$fit`)

## Installation

``` r

# From CRAN
install.packages("tidylearn")

# Development version
# devtools::install_github("ces0491/tidylearn") # nolint
```

``` r

library(tidylearn)
library(dplyr)
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
```

## The Unified Interface

The core of tidylearn is the
[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
function, which dispatches to the appropriate underlying package based
on the method you specify. The wrapped packages include stats, glmnet,
randomForest, xgboost, gbm, e1071, nnet, rpart, cluster, and dbscan.

### Supervised Learning

#### Classification

Logistic regression handles two-class problems, so we take a binary
subset of iris here. For three or more classes use `"tree"`, `"forest"`,
`"svm"` or `"nn"`.

``` r

# versicolor and virginica overlap, so this is a real classification
# problem -- setosa is linearly separable from the other two, which makes
# logistic regression fail to converge
iris_binary <- iris %>%
  filter(Species %in% c("versicolor", "virginica")) %>%
  mutate(Species = droplevels(Species))

model_logistic <- tl_model(iris_binary, Species ~ ., method = "logistic")
print(model_logistic)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: logistic 
#> Task: Classification 
#> Formula: Species ~ . 
#> 
#> Training observations: 100
```

Predictions come back as a tibble with a `.pred` column. What `.pred`
contains depends on `type`: `"class"` gives the predicted label,
`"prob"` gives one column per class.

``` r

# Predicted class labels
predictions <- predict(model_logistic, type = "class")
head(predictions)
#> # A tibble: 6 × 1
#>   .pred     
#>   <fct>     
#> 1 versicolor
#> 2 versicolor
#> 3 versicolor
#> 4 versicolor
#> 5 versicolor
#> 6 versicolor
```

``` r

# Class probabilities
head(predict(model_logistic, type = "prob"))
#> # A tibble: 6 × 2
#>   versicolor virginica
#>        <dbl>     <dbl>
#> 1      1.000 0.0000117
#> 2      1.000 0.0000486
#> 3      0.999 0.00120  
#> 4      1.000 0.0000422
#> 5      0.999 0.00141  
#> 6      1.000 0.000102
```

Note that the default `type = "response"` means different things across
methods — probabilities for logistic regression, class labels for trees
and forests. Ask for `type = "class"` explicitly when you want labels,
or let
[`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
handle it:

``` r

tl_evaluate(model_logistic, metrics = c("accuracy", "f1"))
#> # A tibble: 2 × 2
#>   metric   value
#>   <chr>    <dbl>
#> 1 accuracy  0.98
#> 2 f1        0.98
```

#### Regression

``` r

# Regression with linear model
model_linear <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")
print(model_linear)
#> tidylearn Model
#> ===============
#> Paradigm: supervised 
#> Method: linear 
#> Task: Regression 
#> Formula: mpg ~ wt + hp 
#> 
#> Training observations: 32
```

``` r

# Predictions
predictions_reg <- predict(model_linear)
head(predictions_reg)
#> # A tibble: 6 × 1
#>   .pred
#>   <dbl>
#> 1  23.6
#> 2  22.6
#> 3  25.3
#> 4  21.3
#> 5  18.3
#> 6  20.5
```

### Unsupervised Learning

#### Dimensionality Reduction

``` r

# Principal Component Analysis
model_pca <- tl_model(iris[, 1:4], method = "pca")
print(model_pca)
#> tidylearn Model
#> ===============
#> Paradigm: unsupervised 
#> Method: pca 
#> Technique: pca 
#> 
#> Training observations: 150
```

``` r

# Transform data
transformed <- predict(model_pca)
head(transformed)
#> # A tibble: 6 × 5
#>   .obs_id   PC1    PC2     PC3      PC4
#>   <chr>   <dbl>  <dbl>   <dbl>    <dbl>
#> 1 1       -2.26 -0.478  0.127   0.0241 
#> 2 2       -2.07  0.672  0.234   0.103  
#> 3 3       -2.36  0.341 -0.0441  0.0283 
#> 4 4       -2.29  0.595 -0.0910 -0.0657 
#> 5 5       -2.38 -0.645 -0.0157 -0.0358 
#> 6 6       -2.07 -1.48  -0.0269  0.00659
```

#### Clustering

``` r

# K-means clustering
model_kmeans <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
print(model_kmeans)
#> tidylearn Model
#> ===============
#> Paradigm: unsupervised 
#> Method: kmeans 
#> Technique: kmeans 
#> 
#> Training observations: 150
```

``` r

# Get cluster assignments
clusters <- model_kmeans$fit$clusters
head(clusters)
#> # A tibble: 6 × 2
#>   .obs_id cluster
#>   <chr>     <int>
#> 1 1             1
#> 2 2             1
#> 3 3             1
#> 4 4             1
#> 5 5             1
#> 6 6             1
```

``` r

# Compare with actual species
table(clusters$cluster, iris$Species)
#>    
#>     setosa versicolor virginica
#>   1     50          0         0
#>   2      0         48        14
#>   3      0          2        36
```

## Data Preprocessing

[`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
handles imputation, scaling and encoding in one call, and records what
it did so the same transformation can be replayed on new data:

``` r

# Prepare data with multiple preprocessing steps
processed <- tl_prepare_data(
  iris,
  Species ~ .,
  impute_method = "mean",
  scale_method = "standardize",
  encode_categorical = FALSE
)
#> Scaling numeric features using method: standardize
```

``` r

# Check preprocessing steps applied
names(processed$preprocessing_steps)
#> [1] "scaling"
```

``` r

# Use processed data for modeling
model_processed <- tl_model(processed$data, Species ~ ., method = "forest")
```

## Train-Test Splitting

``` r

# Simple random split
split <- tl_split(iris, prop = 0.7, seed = 123)

# Train model (three species, so a multiclass-capable method)
model_train <- tl_model(split$train, Species ~ ., method = "forest")

# Test predictions
predictions_test <- predict(model_train, new_data = split$test)
head(predictions_test)
#> # A tibble: 6 × 1
#>   .pred 
#>   <fct> 
#> 1 setosa
#> 2 setosa
#> 3 setosa
#> 4 setosa
#> 5 setosa
#> 6 setosa
```

``` r

# Stratified split (maintains class proportions)
split_strat <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 123)

# Check proportions are maintained
prop.table(table(split_strat$train$Species))
#> 
#>     setosa versicolor  virginica 
#>  0.3333333  0.3333333  0.3333333
prop.table(table(split_strat$test$Species))
#> 
#>     setosa versicolor  virginica 
#>  0.3333333  0.3333333  0.3333333
prop.table(table(iris$Species))
#> 
#>     setosa versicolor  virginica 
#>  0.3333333  0.3333333  0.3333333
```

## Wrapped Packages

tidylearn provides a unified interface to these established R packages:

### Supervised Methods

| Method | Underlying Package | Function Called |
|----|----|----|
| `"linear"` | stats | [`lm()`](https://rdrr.io/r/stats/lm.html) |
| `"polynomial"` | stats | [`lm()`](https://rdrr.io/r/stats/lm.html) with [`poly()`](https://rdrr.io/r/stats/poly.html) |
| `"logistic"` | stats | `glm(..., family = binomial)` |
| `"ridge"`, `"lasso"`, `"elastic_net"` | glmnet | `glmnet()` |
| `"tree"` | rpart | `rpart()` |
| `"forest"` | randomForest | `randomForest()` |
| `"boost"` | gbm | `gbm()` |
| `"xgboost"` | xgboost | `xgb.train()` |
| `"svm"` | e1071 | `svm()` |
| `"nn"` | nnet | `nnet()` |
| `"deep"` | keras | `keras_model_sequential()` |

### Unsupervised Methods

| Method | Underlying Package | Function Called |
|----|----|----|
| `"pca"` | stats | [`prcomp()`](https://rdrr.io/r/stats/prcomp.html) |
| `"mds"` | stats, MASS, smacof | [`cmdscale()`](https://rdrr.io/r/stats/cmdscale.html), `isoMDS()`, etc. |
| `"kmeans"` | stats | [`kmeans()`](https://rdrr.io/r/stats/kmeans.html) |
| `"pam"` | cluster | `pam()` |
| `"clara"` | cluster | `clara()` |
| `"hclust"` | stats | [`hclust()`](https://rdrr.io/r/stats/hclust.html) |
| `"dbscan"` | dbscan | `dbscan()` |

### Accessing the Underlying Model

You always have access to the raw model from the underlying package via
`$fit`:

``` r

# Example: Access the raw randomForest object
model_forest <- tl_model(iris, Species ~ ., method = "forest")
class(model_forest$fit)  # This is the randomForest object
#> [1] "randomForest.formula" "randomForest"

# Use package-specific functions if needed
# randomForest::varImpPlot(model_forest$fit) # nolint
```

## Next Steps

- [`vignette("data-ingestion")`](https://tidylearn.sheetsolved.com/articles/data-ingestion.md)
  — reading from files, databases and cloud sources
- [`vignette("supervised-learning")`](https://tidylearn.sheetsolved.com/articles/supervised-learning.md)
  — classification and regression in depth, and how to replay
  preprocessing on a test set
- [`vignette("unsupervised-learning")`](https://tidylearn.sheetsolved.com/articles/unsupervised-learning.md)
  — clustering, ordination, and choosing the number of clusters
- [`vignette("market-basket")`](https://tidylearn.sheetsolved.com/articles/market-basket.md)
  — association rules
- [`vignette("tuning-and-pipelines")`](https://tidylearn.sheetsolved.com/articles/tuning-and-pipelines.md)
  — hyperparameter search, and bundling a workflow you can save
- [`vignette("automl")`](https://tidylearn.sheetsolved.com/articles/automl.md)
  — searching across methods under a time budget
- [`vignette("diagnostics")`](https://tidylearn.sheetsolved.com/articles/diagnostics.md)
  — assumptions, influence and model comparison
- [`vignette("reporting")`](https://tidylearn.sheetsolved.com/articles/reporting.md)
  — plots and formatted `gt` tables
- [`vignette("integration-workflows")`](https://tidylearn.sheetsolved.com/articles/integration-workflows.md)
  — combining supervised and unsupervised steps
- [`vignette("compute-backends")`](https://tidylearn.sheetsolved.com/articles/compute-backends.md)
  — when a fit is too slow or too large for this machine: GPU routing,
  cost estimates, and the cloud safety model

## Summary

tidylearn is a **wrapper package** that provides:

- **Unified Interface**: One function
  ([`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md))
  that dispatches to proven packages like glmnet, randomForest, xgboost,
  e1071, and others
- **Transparency**: Access raw model objects via `model$fit` for
  package-specific functionality
- **Tidy Output**: All results are tibbles for easy manipulation with
  dplyr and ggplot2
- **Consistent Visualization**: Unified ggplot2-based plots regardless
  of model type

The underlying algorithms are unchanged - tidylearn simply makes them
easier to use together.

``` r

# Quick example combining everything
data_split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)

# Random forests are scale-invariant, so no scaling is needed here. When a
# method does need scaled inputs, the same transformation has to be applied
# to the test set -- see the Supervised Learning vignette.
model_final <- tl_model(data_split$train, Species ~ ., method = "forest")
test_preds <- predict(model_final, new_data = data_split$test)

accuracy <- mean(test_preds$.pred == data_split$test$Species)
cat("Test accuracy:", round(accuracy * 100, 1), "%\n")
#> Test accuracy: 93.3 %
```
