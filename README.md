# tidylearn: A Unified Tidy Approach to Machine Learning

[![CRAN](https://img.shields.io/badge/CRAN-not_yet_published-orange)](https://cran.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`tidylearn` is a comprehensive machine learning package for R that **thoughtfully unifies supervised and unsupervised learning** under a consistent, tidy interface. Unlike packages that simply combine functionalities, tidylearn is designed from the ground up to enable seamless integration between different learning paradigms.

The package brings together:
- **Supervised Learning** (classification & regression)
- **Unsupervised Learning** (clustering & dimensionality reduction)
- **Powerful Integration** (semi-supervised learning, transfer learning, feature engineering)
- **Unified Workflows** (AutoML, comprehensive pipelines, exploratory analysis)

Built on tidyverse principles with extensive support for model pipelines, cross-validation, visualization, and interpretation.

## Key Features

### 🎯 Unified Interface

A single `tl_model()` function works with both supervised and unsupervised methods:

```r
# Supervised: Classification
model <- tl_model(iris, Species ~ ., method = "forest")

# Supervised: Regression
model <- tl_model(mtcars, mpg ~ wt + hp, method = "linear")

# Unsupervised: PCA
model <- tl_model(iris, ~ ., method = "pca")

# Unsupervised: Clustering
model <- tl_model(iris, method = "kmeans", k = 3)
```

### 🔗 Thoughtful Integration

tidylearn isn't just tidysl + tidyul bolted together. It provides powerful integration functions that showcase the synergy between learning paradigms:

#### Dimensionality Reduction as Preprocessing
```r
# Reduce dimensions before supervised learning
reduced <- tl_reduce_dimensions(iris, response = "Species",
                                method = "pca", n_components = 3)
model <- tl_model(reduced$data, Species ~ ., method = "logistic")
```

#### Cluster-Based Feature Engineering
```r
# Add cluster features to boost supervised performance
data_clustered <- tl_add_cluster_features(iris, response = "Species",
                                         method = "kmeans", k = 3)
model <- tl_model(data_clustered, Species ~ ., method = "forest")
```

#### Semi-Supervised Learning
```r
# Train with limited labels using cluster-based label propagation
labeled_idx <- sample(nrow(iris), 15)  # Only 10% labeled!
model <- tl_semisupervised(iris, Species ~ .,
                          labeled_indices = labeled_idx,
                          cluster_method = "kmeans",
                          supervised_method = "logistic")
```

#### Anomaly-Aware Modeling
```r
# Detect and handle outliers before supervised learning
model <- tl_anomaly_aware(data, response ~ .,
                         response = "response",
                         anomaly_method = "dbscan",
                         action = "flag")
```

#### Stratified Models
```r
# Train cluster-specific models for heterogeneous data
models <- tl_stratified_models(mtcars, mpg ~ .,
                              cluster_method = "kmeans", k = 3,
                              supervised_method = "linear")
```

### 🤖 Auto ML

Automated machine learning that explores multiple approaches:

```r
result <- tl_auto_ml(iris, Species ~ .,
                    use_reduction = TRUE,
                    use_clustering = TRUE,
                    time_budget = 300)

# View leaderboard
result$leaderboard

# Get best model
best_model <- result$best_model
```

### 🔍 Exploratory Data Analysis

Comprehensive unsupervised analysis workflow:

```r
eda <- tl_explore(iris, response = "Species")

# Automatically performs:
# - PCA analysis
# - Optimal cluster detection
# - Multiple clustering methods
# - Distance analysis

plot(eda)  # Generate comprehensive visualizations
```

### 📊 Unified Preprocessing

```r
processed <- tl_prepare_data(data, formula,
                            impute_method = "mean",
                            scale_method = "standardize",
                            encode_categorical = TRUE,
                            remove_correlated = TRUE)

model <- tl_model(processed$data, formula, method = "forest")
```

## Installation

```r
# Install from GitHub (when available)
# devtools::install_github("ces0491/tidylearn")

# For now, install from source
devtools::install_local("path/to/tidylearn")
```

## Complete Example: Integrative Workflow

Here's an example showcasing tidylearn's unified approach:

```r
library(tidylearn)

# 1. Exploratory Analysis
eda <- tl_explore(iris, response = "Species")
# Discovers optimal k=3 clusters

# 2. Preprocessing with dimensionality reduction
prepared <- tl_reduce_dimensions(iris, response = "Species",
                                method = "pca", n_components = 5)

# 3. Add cluster features
data_enriched <- tl_add_cluster_features(prepared$data,
                                        response = "Species",
                                        method = "kmeans", k = 3)

# 4. Train supervised model
model <- tl_model(data_enriched, Species ~ ., method = "forest")

# 5. Evaluate
metrics <- tl_evaluate(model)
print(metrics)

# 6. Or just use Auto ML for all of this!
auto_result <- tl_auto_ml(iris, Species ~ .)
```

## Supported Methods

### Supervised Learning

**Classification & Regression:**
- Linear & Polynomial Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Gradient Boosting
- XGBoost
- Support Vector Machines (SVM)
- Neural Networks
- Deep Learning (Keras/TensorFlow)

**Regularization:**
- Ridge Regression
- LASSO
- Elastic Net

### Unsupervised Learning

**Dimensionality Reduction:**
- Principal Component Analysis (PCA)
- Multidimensional Scaling (MDS)
  - Classical MDS
  - Metric MDS
  - Non-metric MDS
  - Sammon mapping
  - Kruskal's MDS

**Clustering:**
- K-Means
- K-Medoids (PAM)
- CLARA (for large datasets)
- Hierarchical Clustering
- DBSCAN (density-based)

**Pattern Mining:**
- Market Basket Analysis (Apriori)

## Package Architecture

tidylearn is organized into thoughtfully integrated modules:

- **core.R**: Unified model interface and base classes
- **preprocessing.R**: Data preparation and feature engineering
- **integration.R**: Functions combining supervised & unsupervised learning
- **workflows.R**: High-level workflows (AutoML, EDA, transfer learning)
- **supervised-*.R**: Supervised learning implementations
- **unsupervised-*.R**: Unsupervised learning implementations
- **pipeline.R**: Advanced modeling pipelines
- **visualization.R**: Unified plotting functions
- **metrics.R**: Evaluation metrics for all paradigms

## Philosophy

tidylearn is built on three core principles:

1. **Unification**: One consistent interface across all learning paradigms
2. **Integration**: Meaningful connections between supervised and unsupervised methods
3. **Practicality**: Real-world workflows that combine multiple techniques

This isn't just tidysl and tidyul combined—it's a thoughtfully orchestrated machine learning toolkit that enables powerful workflows impossible with either package alone.

## Documentation

Comprehensive documentation and vignettes coming soon!

```r
# View package help
?tidylearn

# Explore main functions
?tl_model
?tl_auto_ml
?tl_explore
?tl_reduce_dimensions
?tl_semisupervised
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Author

Cesaire Tobias (cesaire@sheetsolved.com)

## Acknowledgments

tidylearn builds upon and integrates:
- **tidysl**: Tidy supervised learning
- **tidyul**: Tidy unsupervised learning
- The tidyverse ecosystem
- Excellent machine learning packages: glmnet, randomForest, xgboost, cluster, dbscan, and many more

---

**tidylearn**: Where supervised meets unsupervised in perfect harmony.
