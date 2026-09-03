# Reporting with tidylearn

## Overview

Two families turn a fitted model into something publishable.
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) and the
`tl_plot_*()` functions return ggplot2 objects;
[`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
and the `tl_table_*()` functions return `gt` tables. Both dispatch on
model type, so the same call covers a forest and a lasso fit, and both
hand back an object you can keep editing rather than printed output you
cannot.

``` r

library(tidylearn)
library(dplyr)
library(ggplot2)
library(gt)
```

## Plots

[`plot()`](https://rdrr.io/r/graphics/plot.default.html) picks the
visualisation from the model type, and `type` narrows it further where a
model supports more than one view.

### Regression

``` r

model_reg <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

# Actual vs predicted — one call
plot(model_reg, type = "actual_predicted")
```

![](reporting_files/figure-html/plot-regression-1.png)

### Classification

``` r

split <- tl_split(iris, prop = 0.7, stratify = "Species", seed = 42)
model_clf <- tl_model(split$train, Species ~ ., method = "forest")

plot(model_clf, type = "confusion")
```

![](reporting_files/figure-html/plot-classification-1.png)

### PCA

``` r

pca <- tidy_pca(USArrests, scale = TRUE)

tidy_pca_screeplot(pca)
```

![](reporting_files/figure-html/plot-pca-1.png)

``` r

tidy_pca_biplot(pca, label_obs = TRUE)
```

![](reporting_files/figure-html/plot-pca-2.png)

### Regularisation

``` r

model_lasso <- tl_model(mtcars, mpg ~ ., method = "lasso")

tl_plot_regularization_path(model_lasso)
```

![](reporting_files/figure-html/plot-lasso-1.png)

``` r

tl_plot_regularization_cv(model_lasso)
```

![](reporting_files/figure-html/plot-lasso-2.png)

## Tables

[`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
mirrors the plot interface, dispatching on model type and an optional
`type`:

``` r

tl_table(model)                       # auto-selects the best table type
tl_table(model, type = "coefficients") # specific type
```

### Evaluation Metrics

``` r

tl_table_metrics(model_reg)
```

| Model Evaluation Metrics                                    |        |
|-------------------------------------------------------------|--------|
| Metric                                                      | Value  |
| Rmse                                                        | 2.4689 |
| Mae                                                         | 1.9015 |
| Rsq                                                         | 0.8268 |
| tidylearn \| linear (regression) \| mpg ~ wt + hp \| n = 32 |        |

### Coefficients

For linear and logistic models, the table includes standard errors, test
statistics, and p-values, with significant terms highlighted:

``` r

tl_table_coefficients(model_reg)
```

| Linear Model Coefficients |  |  |  |  |  |
|----|----|----|----|----|----|
| Term | Estimate | Std. Error | t value | p |  |
| (Intercept) | 37.2273 | 1.5988 | 23.2847 | 2.57 × 10⁻²⁰ | \* |
| wt | −3.8778 | 0.6327 | −6.1287 | 1.12 × 10⁻⁶ | \* |
| hp | −0.0318 | 0.0090 | −3.5187 | 1.45 × 10⁻³ | \* |
| tidylearn \| linear (regression) \| mpg ~ wt + hp \| n = 32 |  |  |  |  |  |

`conf_int = TRUE` adds a confidence interval, and `level` sets its
width:

``` r

tl_table_coefficients(model_reg, conf_int = TRUE, level = 0.9)
```

| Linear Model Coefficients |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Wald 90% intervals |  |  |  |  |  |  |  |
| Term | Estimate | Std. Error | Lower 90% | Upper 90% | t value | p |  |
| (Intercept) | 37.2273 | 1.5988 | 34.5107 | 39.9438 | 23.2847 | 2.57 × 10⁻²⁰ | \* |
| wt | −3.8778 | 0.6327 | −4.9529 | −2.8027 | −6.1287 | 1.12 × 10⁻⁶ | \* |
| hp | −0.0318 | 0.0090 | −0.0471 | −0.0164 | −3.5187 | 1.45 × 10⁻³ | \* |
| tidylearn \| linear (regression) \| mpg ~ wt + hp \| n = 32 |  |  |  |  |  |  |  |

These are Wald intervals, built from the standard errors in the column
beside them, so the interval and the p-value in a row always agree about
whether zero is excluded. For a logistic model, `exponentiate = TRUE`
reports odds ratios instead of log odds.

For regularised models, coefficients are sorted by magnitude and zero
coefficients are greyed out. There is no interval to add — glmnet
reports no standard errors, and `conf_int = TRUE` is an error here
rather than a column of `NA`:

``` r

tl_table_coefficients(model_lasso)
```

| Lasso Coefficients |  |  |
|----|----|----|
| lambda = 1.399 (1se) |  |  |
| Term | Coefficient | \|Coefficient\| |
| (Intercept) | 33.9411 | 33.9411 |
| wt | −2.3645 | 2.3645 |
| cyl | −0.8434 | 0.8434 |
| hp | −0.0070 | 0.0070 |
| disp | 0.0000 | 0.0000 |
| drat | 0.0000 | 0.0000 |
| qsec | 0.0000 | 0.0000 |
| vs | 0.0000 | 0.0000 |
| am | 0.0000 | 0.0000 |
| gear | 0.0000 | 0.0000 |
| carb | 0.0000 | 0.0000 |
| tidylearn \| lasso (regression) \| mpg ~ . \| n = 32 |  |  |

The numbers behind these tables come from
[`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md),
which takes the same arguments, returns a tibble, and does not need `gt`
installed:

``` r

tl_coefficients(model_reg, conf_int = TRUE)
#> # A tibble: 3 × 7
#>   term        estimate std_error conf_low conf_high statistic  p_value
#>   <chr>          <dbl>     <dbl>    <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)  37.2      1.60     34.0      40.5        23.3  2.57e-20
#> 2 wt           -3.88     0.633    -5.17     -2.58       -6.13 1.12e- 6
#> 3 hp           -0.0318   0.00903  -0.0502   -0.0133     -3.52 1.45e- 3
```

### Confusion Matrix

A formatted confusion matrix with correct predictions highlighted on the
diagonal:

``` r

tl_table_confusion(model_clf, new_data = split$test)
```

[TABLE]

### Feature Importance

A ranked importance table with a colour gradient:

``` r

tl_table_importance(model_clf)
```

| Feature Importance                                             |            |
|----------------------------------------------------------------|------------|
| Top 4 features                                                 |            |
| Feature                                                        | Importance |
| Petal.Length                                                   | 100.00     |
| Petal.Width                                                    | 88.23      |
| Sepal.Length                                                   | 29.63      |
| Sepal.Width                                                    | 15.77      |
| tidylearn \| forest (classification) \| Species ~ . \| n = 105 |            |

### PCA Variance Explained

Cumulative variance is coloured green to highlight how many components
are needed:

``` r

pca_model <- tl_model(USArrests, method = "pca")
tl_table_variance(pca_model)
```

| PCA Variance Explained     |           |          |            |            |
|----------------------------|-----------|----------|------------|------------|
| Component                  | Std. Dev. | Variance | Proportion | Cumulative |
| PC1                        | 1.5749    | 2.4802   | 62.0%      | 62.0%      |
| PC2                        | 0.9949    | 0.9898   | 24.7%      | 86.8%      |
| PC3                        | 0.5971    | 0.3566   | 8.9%       | 95.7%      |
| PC4                        | 0.4164    | 0.1734   | 4.3%       | 100.0%     |
| tidylearn \| pca \| n = 50 |           |          |            |            |

### PCA Loadings

A diverging red–blue colour scale highlights strong positive and
negative loadings:

``` r

tl_table_loadings(pca_model)
```

| PCA Loadings               |        |        |        |        |
|----------------------------|--------|--------|--------|--------|
| Variable                   | PC1    | PC2    | PC3    | PC4    |
| Murder                     | −0.536 | −0.418 | 0.341  | 0.649  |
| Assault                    | −0.583 | −0.188 | 0.268  | −0.743 |
| UrbanPop                   | −0.278 | 0.873  | 0.378  | 0.134  |
| Rape                       | −0.543 | 0.167  | −0.818 | 0.089  |
| tidylearn \| pca \| n = 50 |        |        |        |        |

### Cluster Summary

Cluster sizes and mean feature values:

``` r

km <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
tl_table_clusters(km)
```

| Cluster Summary |  |  |  |  |  |
|----|----|----|----|----|----|
| kmeans \| 3 clusters |  |  |  |  |  |
| Cluster | Size | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |
| 1 | 38 | 6.85 | 3.07 | 5.74 | 2.07 |
| 2 | 50 | 5.01 | 3.43 | 1.46 | 0.25 |
| 3 | 62 | 5.90 | 2.75 | 4.39 | 1.43 |
| tidylearn \| kmeans \| n = 150 |  |  |  |  |  |

### Model Comparison

Compare multiple models side-by-side:

``` r

m1 <- tl_model(split$train, Species ~ ., method = "svm")
m2 <- tl_model(split$train, Species ~ ., method = "forest")
m3 <- tl_model(split$train, Species ~ ., method = "tree")

tl_table_comparison(
  m1, m2, m3,
  new_data = split$test,
  names = c("SVM", "Random Forest", "Decision Tree")
)
```

| Model Comparison    |        |               |               |
|---------------------|--------|---------------|---------------|
| 3 models compared   |        |               |               |
| Metric              | SVM    | Random Forest | Decision Tree |
| Accuracy            | 0.9111 | 0.9333        | 0.8889        |
| tidylearn \| n = 45 |        |               |               |

## Interactive Reporting with plotly

Every plot function returns a ggplot2 object, so
[`ggplotly()`](https://rdrr.io/pkg/plotly/man/ggplotly.html) takes any
of them without special handling:

``` r

library(plotly)

ggplotly(plot(model_reg, type = "actual_predicted"))
ggplotly(tidy_pca_biplot(pca, label_obs = TRUE))
ggplotly(tl_plot_regularization_path(model_lasso))
```

## Putting It Together

Fit, score, look, drill in — the four calls that make up most reporting
sections:

``` r

# Fit
model <- tl_model(split$train, Species ~ ., method = "forest")

# Evaluate
tl_table_metrics(model, new_data = split$test)
```

| Model Evaluation Metrics                                       |        |
|----------------------------------------------------------------|--------|
| Metric                                                         | Value  |
| Accuracy                                                       | 0.9333 |
| tidylearn \| forest (classification) \| Species ~ . \| n = 105 |        |

``` r


# Visualise
plot(model, type = "confusion")
```

![](reporting_files/figure-html/workflow-1.png)

``` r


# Drill into feature importance
tl_table_importance(model, top_n = 4)
```

| Feature Importance                                             |            |
|----------------------------------------------------------------|------------|
| Top 4 features                                                 |            |
| Feature                                                        | Importance |
| Petal.Length                                                   | 100.00     |
| Petal.Width                                                    | 91.09      |
| Sepal.Length                                                   | 28.28      |
| Sepal.Width                                                    | 13.44      |
| tidylearn \| forest (classification) \| Species ~ . \| n = 105 |            |

Swap `method = "forest"` for `method = "tree"` or `method = "svm"` and
the reporting code above works without modification.
