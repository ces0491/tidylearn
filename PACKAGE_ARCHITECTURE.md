# tidylearn Package Architecture

## Design Philosophy

tidylearn is **not** simply tidysl and tidyul combined. It is a thoughtfully orchestrated machine learning framework that enables seamless integration between supervised and unsupervised learning paradigms.

### Core Principles

1. **Unification**: Single consistent interface (`tl_model()`) for all learning methods
2. **Integration**: Meaningful connections between supervised and unsupervised techniques
3. **Practicality**: Real-world workflows that combine multiple learning paradigms

## Package Structure

### Core Modules

#### `core.R` - Unified Model Interface
- **`tl_model()`**: Single entry point for all modeling
- Automatic routing to supervised/unsupervised implementations
- Base class hierarchy: `tidylearn_model` → `tidylearn_supervised` / `tidylearn_unsupervised`
- Unified S3 methods: `print()`, `summary()`, `predict()`, `plot()`

**Key Innovation**: Seamlessly handles both supervised and unsupervised models with the same function call.

```r
# Same interface, different paradigms
supervised <- tl_model(iris, Species ~ ., method = "forest")
unsupervised <- tl_model(iris, method = "kmeans", k = 3)
```

#### `integration.R` - The Heart of Unification
This is where tidylearn truly shines. Functions that meaningfully combine learning paradigms:

- **`tl_reduce_dimensions()`**: Use PCA/MDS as preprocessing for supervised learning
- **`tl_add_cluster_features()`**: Semi-supervised feature engineering
- **`tl_semisupervised()`**: Train with limited labels via cluster-based label propagation
- **`tl_anomaly_aware()`**: Detect outliers before supervised modeling
- **`tl_stratified_models()`**: Cluster-specific supervised models for heterogeneous data

**Key Innovation**: These functions represent workflows impossible with either tidysl or tidyul alone.

#### `workflows.R` - High-Level Intelligent Workflows

- **`tl_auto_ml()`**: Automated ML that explores:
  - Baseline supervised models
  - Models with dimensionality reduction
  - Models with cluster features
  - Advanced models
  - Returns best model with leaderboard

- **`tl_explore()`**: Comprehensive unsupervised EDA
  - PCA analysis
  - Optimal cluster detection
  - Multiple clustering methods
  - Integrated visualizations

- **`tl_transfer_learning()`**: Unsupervised pre-training → supervised fine-tuning

**Key Innovation**: Intelligent workflows that automatically apply integration techniques.

#### `preprocessing.R` - Unified Data Preparation

- **`tl_prepare_data()`**: Comprehensive preprocessing
  - Missing value imputation
  - Categorical encoding
  - Feature scaling
  - Zero-variance removal
  - Correlation-based feature removal

- **`tl_split()`**: Train-test splitting with stratification

**Key Innovation**: Works seamlessly with both supervised and unsupervised workflows.

### Supervised Learning Modules

All adapted from tidysl with consistent naming:

- `supervised-classification.R`: Logistic regression
- `supervised-regression.R`: Linear, polynomial regression
- `supervised-regularization.R`: Ridge, LASSO, Elastic Net
- `supervised-trees.R`: Decision trees, Random Forests, Gradient Boosting
- `supervised-svm.R`: Support Vector Machines
- `supervised-neural-networks.R`: Basic neural networks
- `supervised-deep-learning.R`: Keras/TensorFlow deep learning
- `supervised-xgboost.R`: XGBoost implementation

### Unsupervised Learning Modules

All adapted from tidyul with consistent naming:

- `unsupervised-pca.R`: Principal Component Analysis
- `unsupervised-mds.R`: Multidimensional Scaling
- `unsupervised-clustering.R`: K-Means, PAM, CLARA
- `unsupervised-hclust.R`: Hierarchical clustering
- `unsupervised-dbscan.R`: Density-based clustering
- `unsupervised-market-basket.R`: Apriori algorithm
- `unsupervised-distance.R`: Distance metrics
- `unsupervised-validation.R`: Cluster validation

### Supporting Modules

- `pipeline.R`: Advanced modeling pipelines (from tidysl)
- `model-selection.R`: Cross-validation, model comparison
- `tuning.R`: Hyperparameter tuning
- `interactions.R`: Interaction effects analysis
- `diagnostics.R`: Model diagnostics
- `metrics.R`: Evaluation metrics
- `visualization.R`: Unified plotting (combined from both packages)
- `utils.R`: Helper functions

## Integration Points

### 1. Feature Engineering
```r
# Unsupervised → Supervised
reduced <- tl_reduce_dimensions(data, response = "y", method = "pca")
model <- tl_model(reduced$data, y ~ ., method = "forest")
```

### 2. Semi-Supervised Learning
```r
# Clustering + Supervised
model <- tl_semisupervised(data, y ~ .,
                          labeled_indices = labeled_idx,
                          cluster_method = "kmeans",
                          supervised_method = "logistic")
```

### 3. Anomaly Detection
```r
# Unsupervised outlier detection → Supervised learning
model <- tl_anomaly_aware(data, y ~ .,
                         anomaly_method = "dbscan",
                         action = "flag")
```

### 4. Stratified Modeling
```r
# Cluster → Multiple supervised models
models <- tl_stratified_models(data, y ~ .,
                               cluster_method = "kmeans", k = 3,
                               supervised_method = "linear")
```

### 5. Transfer Learning
```r
# Unsupervised pre-training → Supervised fine-tuning
model <- tl_transfer_learning(data, y ~ .,
                              pretrain_method = "pca",
                              supervised_method = "logistic")
```

## Class Hierarchy

```
tidylearn_model (base class)
├── tidylearn_supervised
│   ├── tidylearn_linear
│   ├── tidylearn_logistic
│   ├── tidylearn_forest
│   ├── tidylearn_xgboost
│   └── ... (other supervised methods)
├── tidylearn_unsupervised
│   ├── tidylearn_pca
│   ├── tidylearn_kmeans
│   ├── tidylearn_hclust
│   └── ... (other unsupervised methods)
├── tidylearn_semisupervised (integration)
├── tidylearn_stratified (integration)
├── tidylearn_transfer (integration)
├── tidylearn_automl (workflow)
└── tidylearn_eda (workflow)
```

## Function Naming Convention

- `tl_model()`: Create any model
- `tl_*()`: High-level user-facing functions
- `tl_fit_*()`: Internal fitting functions (method-specific)
- `tl_predict_*()`: Internal prediction functions (method-specific)
- `tl_plot_*()`: Visualization functions

## Key Differentiators from tidysl + tidyul

### What Makes tidylearn Unique:

1. **Unified Interface**: One function (`tl_model()`) for all paradigms
2. **Integration Functions**: `tl_reduce_dimensions()`, `tl_semisupervised()`, `tl_anomaly_aware()`, etc.
3. **Intelligent Workflows**: `tl_auto_ml()`, `tl_explore()`, `tl_transfer_learning()`
4. **Seamless Preprocessing**: Works with both paradigms
5. **Combined Visualizations**: Unified plotting across all methods
6. **Practical Patterns**: Real-world ML workflows built-in

### Not Just Combining Packages:

tidysl focuses on:
- Supervised learning only
- `tl_` prefix for supervised methods
- Separate interfaces for classification/regression

tidyul focuses on:
- Unsupervised learning only
- `tidy_` prefix for unsupervised methods
- Standalone clustering/dimensionality reduction

**tidylearn provides**:
- **Both paradigms under one roof**
- **Meaningful integration between them**
- **Workflows that leverage both**
- **Consistent interface across everything**
- **Advanced patterns (semi-supervised, transfer learning, etc.)**

## Usage Patterns

### Pattern 1: Pure Supervised
```r
model <- tl_model(data, y ~ ., method = "forest")
```

### Pattern 2: Pure Unsupervised
```r
clusters <- tl_model(data, method = "kmeans", k = 3)
```

### Pattern 3: Integrated (The tidylearn Way)
```r
# Explore → Reduce → Engineer → Model
eda <- tl_explore(data, response = "y")
reduced <- tl_reduce_dimensions(data, response = "y", method = "pca")
enriched <- tl_add_cluster_features(reduced$data, response = "y")
model <- tl_model(enriched, y ~ ., method = "forest")
```

### Pattern 4: Auto ML (Let tidylearn do it)
```r
result <- tl_auto_ml(data, y ~ .)
best_model <- result$best_model
```

## File Organization

```
tidylearn/
├── DESCRIPTION           # Package metadata
├── NAMESPACE             # Exported functions
├── LICENSE               # MIT License
├── README.md             # User documentation
├── PACKAGE_ARCHITECTURE.md  # This file
├── tidylearn.Rproj       # RStudio project
├── .Rbuildignore         # Build configuration
├── .gitignore            # Git configuration
├── R/
│   ├── core.R                          # Unified interface
│   ├── utils.R                         # Utilities
│   ├── preprocessing.R                 # Data preparation
│   ├── integration.R                   # Integration functions ⭐
│   ├── workflows.R                     # High-level workflows ⭐
│   ├── supervised-*.R                  # Supervised methods (8 files)
│   ├── unsupervised-*.R               # Unsupervised methods (8 files)
│   ├── pipeline.R                      # Advanced pipelines
│   ├── model-selection.R              # Cross-validation
│   ├── tuning.R                       # Hyperparameter tuning
│   ├── interactions.R                 # Interaction effects
│   ├── diagnostics.R                  # Model diagnostics
│   ├── metrics.R                      # Evaluation metrics
│   └── visualization.R                # Unified plotting
├── man/                  # Documentation (auto-generated)
├── tests/                # Unit tests
│   └── testthat/
├── vignettes/            # Long-form documentation
└── inst/
    └── examples/
        └── unified_workflow.R  # Complete examples ⭐
```

⭐ = Unique to tidylearn (not in tidysl or tidyul)

## Dependencies

### Core Dependencies
- tidyverse ecosystem: dplyr, ggplot2, tibble, tidyr, purrr, rlang
- stats, utils (base R)

### Supervised Learning
- glmnet, randomForest, rpart, gbm, e1071, nnet
- ROCR, yardstick (evaluation)

### Unsupervised Learning
- cluster, dbscan, factoextra, MASS

### Optional (Suggests)
- keras, tensorflow (deep learning)
- xgboost (gradient boosting)
- caret, parsnip (integration)
- arules, arulesViz (market basket)
- Various visualization and diagnostic packages

## Development Roadmap

### Completed
✅ Unified model interface
✅ Integration functions
✅ Auto ML workflow
✅ EDA workflow
✅ Preprocessing pipeline
✅ Core supervised methods
✅ Core unsupervised methods
✅ Documentation and examples

### Future Enhancements
- [ ] More integration patterns (ensemble learning)
- [ ] Advanced AutoML features (neural architecture search)
- [ ] Model interpretability (SHAP, LIME)
- [ ] Time series integration
- [ ] Automated feature engineering
- [ ] Production deployment tools
- [ ] Interactive dashboards
- [ ] Comprehensive vignettes

## Conclusion

Instead of treating supervised and unsupervised learning as separate domains, tidylearn recognizes them as complementary techniques that work better together.

The package doesn't just combine tidysl and tidyul—it **reimagines** them as part of a unified machine learning ecosystem.
