# tidylearn <img src="man/figures/logo.png" align="right" height="139" alt="tidylearn logo" />

A Unified Tidy Interface to Machine Learning in R

[![CRAN](https://img.shields.io/badge/CRAN-not_yet_published-orange)](https://cran.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`tidylearn` provides a **unified tidyverse-compatible interface** to R's machine
learning ecosystem. It wraps proven packages like glmnet, randomForest,
xgboost, e1071, cluster, and dbscan - you get the reliability of established
implementations with the convenience of a consistent, tidy API.

**What tidylearn does:**

- Provides one consistent interface (`tl_model()`) to 20+ ML algorithms
- Returns tidy tibbles instead of varied output formats
- Offers unified ggplot2-based visualization across all methods
- Enables pipe-friendly workflows with `%>%`
- Orchestrates complex workflows combining multiple techniques

**What tidylearn is NOT:**

- A reimplementation of ML algorithms (uses established packages under the hood)
- A replacement for the underlying packages (you can access the raw model via
  `model$fit`)

## Why tidylearn?

Each ML package in R has its own API, output format, and conventions. tidylearn
provides a translation layer so you can:

| Without tidylearn                     | With tidylearn          |
| ------------------------------------- | ----------------------- |
| Learn different APIs for each package | One API for everything  |
| Write custom code to extract results  | Consistent tibble output |
| Create different plots for each model | Unified visualization   |
| Manage package-specific quirks        | Focus on your analysis  |

The underlying algorithms are unchanged - tidylearn simply makes them easier to
use together.

## Installation

```r
# Install from GitHub
# devtools::install_github("ces0491/tidylearn")

```

## Quick Start

### Unified Interface

A single `tl_model()` function dispatches to the appropriate underlying package:

```r
library(tidylearn)

# Classification -> uses randomForest::randomForest()
model <- tl_model(iris, Species ~ ., method = "forest")

# Regression -> uses stats::lm()
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

# Regularization -> uses glmnet::glmnet()
model <- tl_model(mtcars, mpg ~ ., method = "lasso")

# Clustering -> uses stats::kmeans()
model <- tl_model(iris[,1:4], method = "kmeans", k = 3)

# PCA -> uses stats::prcomp()
model <- tl_model(iris[,1:4], method = "pca")
```

### Tidy Output

All results come back as tibbles, ready for dplyr and ggplot2:

```r
# Predictions as tibbles
predictions <- predict(model, new_data = test_data)

# Metrics as tibbles
metrics <- tl_evaluate(model, test_data)

# Easy to pipe
model %>%
  predict(test_data) %>%
  bind_cols(test_data) %>%
  ggplot(aes(x = actual, y = prediction)) +
  geom_point()
```

### Access the Underlying Model

You always have access to the raw model from the underlying package:

```r
model <- tl_model(iris, Species ~ ., method = "forest")

# Access the randomForest object directly
model$fit  # This is the randomForest::randomForest() result

# Use package-specific functions if needed
randomForest::varImpPlot(model$fit)
```

## Wrapped Packages

tidylearn provides a unified interface to these established R packages:

### Supervised Learning

| Method | Underlying Package | Function Called |
|--------|-------------------|-----------------|
| `"linear"` | stats | `lm()` |
| `"polynomial"` | stats | `lm()` with `poly()` |
| `"logistic"` | stats | `glm(..., family = binomial)` |
| `"ridge"`, `"lasso"`, `"elastic_net"` | glmnet | `glmnet()` |
| `"tree"` | rpart | `rpart()` |
| `"forest"` | randomForest | `randomForest()` |
| `"boost"` | gbm | `gbm()` |
| `"xgboost"` | xgboost | `xgb.train()` |
| `"svm"` | e1071 | `svm()` |
| `"nn"` | nnet | `nnet()` |
| `"deep"` | keras | `keras_model_sequential()` |

### Unsupervised Learning

| Method | Underlying Package | Function Called |
|--------|-------------------|-----------------|
| `"pca"` | stats | `prcomp()` |
| `"mds"` | stats, MASS, smacof | `cmdscale()`, `isoMDS()`, etc. |
| `"kmeans"` | stats | `kmeans()` |
| `"pam"` | cluster | `pam()` |
| `"clara"` | cluster | `clara()` |
| `"hclust"` | stats | `hclust()` |
| `"dbscan"` | dbscan | `dbscan()` |

## Integration Workflows

Beyond wrapping individual packages, tidylearn provides orchestration functions
that combine multiple techniques:

### Dimensionality Reduction + Supervised Learning

```r
# Reduce dimensions before classification
reduced <- tl_reduce_dimensions(iris, response = "Species",
                                method = "pca", n_components = 3)
model <- tl_model(reduced$data, Species ~ ., method = "logistic")
```

### Cluster-Based Feature Engineering

```r
# Add cluster membership as a feature
enriched <- tl_add_cluster_features(data, response = "target",
                                    method = "kmeans", k = 3)
model <- tl_model(enriched, target ~ ., method = "forest")
```

### Semi-Supervised Learning

```r
# Use clustering to propagate labels to unlabeled data
model <- tl_semisupervised(data, target ~ .,
                          labeled_indices = labeled_idx,
                          cluster_method = "kmeans")
```

### AutoML

```r
# Automatically try multiple approaches
result <- tl_auto_ml(data, target ~ .,
                    time_budget = 300)
result$leaderboard
```

## Unified Visualization

Consistent ggplot2-based plotting regardless of model type:

```r
# All models support similar plotting interface
tl_plot_importance(forest_model)
tl_plot_diagnostics(linear_model)
tl_plot_roc(classification_model)
plot_clusters(clustering_result)
plot_variance_explained(pca_result)
```

## Philosophy

tidylearn is built on these principles:

1. **Transparency**: The underlying packages do the real work. tidylearn makes
   them easier to use together without hiding what's happening.

2. **Consistency**: One interface, tidy output, unified visualization - across
   all methods.

3. **Accessibility**: Focus on your analysis, not on learning different package
   APIs.

4. **Interoperability**: Results work seamlessly with dplyr, ggplot2, and the
   broader tidyverse.

## Documentation

```r
# View package help
?tidylearn

# Explore main functions
?tl_model
?tl_evaluate
?tl_auto_ml
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Author

Cesaire Tobias (<cesaire@sheetsolved.com>)

## Acknowledgments

tidylearn is a wrapper that builds upon the excellent work of many R package
authors. The actual algorithms are implemented in:

- **stats** (base R): lm, glm, prcomp, kmeans, hclust, cmdscale
- **glmnet**: Ridge, LASSO, and elastic net regularization
- **randomForest**: Random forest implementation
- **xgboost**: Gradient boosting
- **gbm**: Gradient boosting machines
- **e1071**: Support vector machines
- **nnet**: Neural networks
- **rpart**: Decision trees
- **cluster**: PAM, CLARA clustering
- **dbscan**: Density-based clustering
- **MASS**: Sammon mapping, isoMDS
- **smacof**: SMACOF MDS algorithm
- **keras/tensorflow**: Deep learning (optional)

Thank you to all the package maintainers whose work makes tidylearn possible.

---
