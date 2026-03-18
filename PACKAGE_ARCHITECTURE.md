# tidylearn Package Architecture

## Design Philosophy

tidylearn is a **wrapper package** that provides a unified tidyverse-compatible
interface to R's machine learning ecosystem. It wraps established packages like
glmnet, randomForest, xgboost, e1071, cluster, and dbscan - you get the
reliability of established implementations with the convenience of a consistent,
tidy API.

**What tidylearn does:**

- Reads data from diverse sources (`tl_read()`) — files, databases, cloud, APIs
- Provides one consistent interface (`tl_model()`) to 20+ ML algorithms
- Returns tidy tibbles instead of varied output formats
- Offers unified ggplot2-based visualization and formatted `gt` tables
- Enables pipe-friendly workflows with `%>%`
- Orchestrates complex workflows combining multiple techniques

**What tidylearn is NOT:**

- A reimplementation of ML algorithms (uses established packages under the hood)
- A replacement for the underlying packages (access raw models via `model$fit`)

### Core Principles

1. **Transparency**: The underlying packages do the real work. tidylearn makes
   them easier to use together without hiding what's happening.
2. **Consistency**: One interface, tidy output, unified visualization - across
   all methods.
3. **Accessibility**: Focus on your analysis, not on learning different package
   APIs.
4. **Interoperability**: Results work seamlessly with dplyr, ggplot2, and the
   broader tidyverse.

## Wrapped Packages

### Supervised Learning

| Method                                 | Underlying Package  | Function Called               |
| -------------------------------------- | ------------------- | ----------------------------- |
| `"linear"`                             | stats               | `lm()`                        |
| `"polynomial"`                         | stats               | `lm()` with `poly()`          |
| `"logistic"`                           | stats               | `glm(..., family = binomial)` |
| `"ridge"`, `"lasso"`, `"elastic_net"`  | glmnet              | `glmnet()`                    |
| `"tree"`                               | rpart               | `rpart()`                     |
| `"forest"`                             | randomForest        | `randomForest()`              |
| `"boost"`                              | gbm                 | `gbm()`                       |
| `"xgboost"`                            | xgboost             | `xgb.train()`                 |
| `"svm"`                                | e1071               | `svm()`                       |
| `"nn"`                                 | nnet                | `nnet()`                      |
| `"deep"`                               | keras               | `keras_model_sequential()`    |

### Unsupervised Learning

| Method      | Underlying Package   | Function Called                 |
| ----------- | -------------------- | ------------------------------- |
| `"pca"`     | stats                | `prcomp()`                      |
| `"mds"`     | stats, MASS, smacof  | `cmdscale()`, `isoMDS()`, etc.  |
| `"kmeans"`  | stats                | `kmeans()`                      |
| `"pam"`     | cluster              | `pam()`                         |
| `"clara"`   | cluster              | `clara()`                       |
| `"hclust"`  | stats                | `hclust()`                      |
| `"dbscan"`  | dbscan               | `dbscan()`                      |

## Package Structure

### Data Ingestion Module

#### `read.R` - Unified Data Reading

- **`tl_read()`**: Single entry point that auto-detects format and dispatches
- **`tl_read_csv()`**, **`tl_read_tsv()`**: Delimited files via readr (base R fallback)
- **`tl_read_excel()`**: Excel files via readxl
- **`tl_read_parquet()`**: Parquet files via nanoparquet
- **`tl_read_json()`**: JSON files via jsonlite
- **`tl_read_rds()`**, **`tl_read_rdata()`**: Native R formats via base R
- **`tl_read_dir()`**: Scan directories for data files
- **`tl_read_zip()`**: Extract and read from zip archives
- Multi-path support: `tl_read(c("a.csv", "b.csv"))` row-binds with `source_file` column
- All readers return `tidylearn_data` objects (tibble subclass with source metadata)

#### `read-backends.R` - Database and Cloud Readers

- **`tl_read_db()`**: Query any live DBI connection
- **`tl_read_sqlite()`**: Auto-connect to SQLite files via RSQLite
- **`tl_read_postgres()`**: PostgreSQL via RPostgres
- **`tl_read_mysql()`**: MySQL/MariaDB via RMariaDB
- **`tl_read_bigquery()`**: Google BigQuery via bigrquery
- **`tl_read_s3()`**: Amazon S3 via paws.storage
- **`tl_read_github()`**: Raw file download from repositories
- **`tl_read_kaggle()`**: Dataset download via Kaggle CLI

All backend packages are suggested dependencies, checked at call time.

### Core Modules

#### `core.R` - Unified Model Interface

- **`tl_model()`**: Single entry point that dispatches to underlying packages
- Automatic routing to supervised/unsupervised implementations
- Unified S3 methods: `print()`, `summary()`, `predict()`, `plot()`
- Raw model accessible via `$fit` slot

```r
# Same interface, different underlying packages
supervised <- tl_model(iris, Species ~ ., method = "forest")
supervised$fit  # Access randomForest::randomForest() result

unsupervised <- tl_model(iris[,1:4], method = "kmeans", k = 3)
unsupervised$fit  # Access stats::kmeans() result
```

#### `integration.R` - Workflow Orchestration

Functions that coordinate multiple wrapped packages:

- **`tl_reduce_dimensions()`**: Use PCA/MDS as preprocessing for supervised
  learning
- **`tl_add_cluster_features()`**: Add cluster membership as features
- **`tl_semisupervised()`**: Cluster-based label propagation
- **`tl_anomaly_aware()`**: Outlier detection before supervised modeling
- **`tl_stratified_models()`**: Cluster-specific supervised models

#### `workflows.R` - High-Level Workflows

- **`tl_auto_ml()`**: Automated ML that tries multiple approaches
- **`tl_explore()`**: Comprehensive unsupervised EDA
- **`tl_transfer_learning()`**: Unsupervised pre-training with supervised
  fine-tuning

#### `preprocessing.R` - Unified Data Preparation

- **`tl_prepare_data()`**: Comprehensive preprocessing
- **`tl_split()`**: Train-test splitting with stratification

### Supervised Learning Modules

Each module wraps a specific package:

- `supervised-classification.R`: Wraps stats::glm for logistic regression
- `supervised-regression.R`: Wraps stats::lm for linear/polynomial regression
- `supervised-regularization.R`: Wraps glmnet for ridge/LASSO/elastic net
- `supervised-trees.R`: Wraps rpart, randomForest, gbm
- `supervised-svm.R`: Wraps e1071::svm
- `supervised-neural-networks.R`: Wraps nnet
- `supervised-deep-learning.R`: Wraps keras (optional)
- `supervised-xgboost.R`: Wraps xgboost (optional)

### Unsupervised Learning Modules

Each module wraps specific packages:

- `unsupervised-pca.R`: Wraps stats::prcomp
- `unsupervised-mds.R`: Wraps stats::cmdscale, MASS::isoMDS, smacof
- `unsupervised-clustering.R`: Wraps stats::kmeans, cluster::pam, cluster::clara
- `unsupervised-hclust.R`: Wraps stats::hclust
- `unsupervised-dbscan.R`: Wraps dbscan::dbscan
- `unsupervised-market-basket.R`: Wraps arules (optional)
- `unsupervised-distance.R`: Distance metric utilities
- `unsupervised-validation.R`: Cluster validation metrics

### Publishing Modules

- `visualization.R`: Unified ggplot2-based plotting — `plot()` dispatches by
  model type, plus specialized plot functions (`tl_plot_*()`)
- `tables.R`: Formatted `gt` tables — `tl_table()` dispatches by model type,
  plus `tl_table_metrics()`, `tl_table_coefficients()`, `tl_table_confusion()`,
  `tl_table_importance()`, `tl_table_variance()`, `tl_table_loadings()`,
  `tl_table_clusters()`, `tl_table_comparison()`

### Supporting Modules

- `pipeline.R`: Advanced modeling pipelines
- `model-selection.R`: Cross-validation, model comparison
- `tuning.R`: Hyperparameter tuning
- `interactions.R`: Interaction effects analysis
- `diagnostics.R`: Model diagnostics
- `metrics.R`: Evaluation metrics
- `utils.R`: Helper functions

## Function Naming Convention

- `tl_read()` / `tl_read_*()`: Data ingestion from any source
- `tl_model()`: Create any model (main user interface)
- `tl_*()`: High-level user-facing functions
- `tl_fit_*()`: Internal fitting functions (call underlying packages)
- `tl_predict_*()`: Internal prediction functions
- `tl_plot_*()`: Visualization functions
- `tl_table()` / `tl_table_*()`: Formatted gt tables

## Usage Patterns

### Pattern 1: Full Pipeline

```r
# Ingest -> Prepare -> Model -> Evaluate -> Publish
data <- tl_read("sales.csv")
split <- tl_split(data, prop = 0.7, stratify = "target")
model <- tl_model(split$train, target ~ ., method = "forest")
tl_evaluate(model, new_data = split$test)
tl_table_metrics(model, new_data = split$test)
```

### Pattern 2: Direct Model Fitting

```r
# Wraps randomForest::randomForest()
model <- tl_model(data, y ~ ., method = "forest")

# Access raw model for package-specific functions
randomForest::varImpPlot(model$fit)
```

### Pattern 3: Workflow Orchestration

```r
# Coordinates multiple packages
eda <- tl_explore(data, response = "y")
reduced <- tl_reduce_dimensions(data, response = "y", method = "pca")
enriched <- tl_add_cluster_features(reduced$data, response = "y")
model <- tl_model(enriched, y ~ ., method = "forest")
```

### Pattern 4: Automated ML

```r
# Tries multiple wrapped packages, returns best
result <- tl_auto_ml(data, y ~ .)
best_model <- result$best_model
```

## File Organization

```
tidylearn/
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
├── README.md
├── NEWS.md
├── cran-comments.md
├── PACKAGE_ARCHITECTURE.md  # This file
├── .lintr                        # Lintr configuration
├── R/
│   ├── utils.R                   # Utilities
│   ├── read.R                    # Data ingestion dispatcher
│   ├── read-backends.R           # Database/cloud readers
│   ├── core.R                    # Unified model interface
│   ├── preprocessing.R           # Data preparation
│   ├── integration.R             # Workflow orchestration
│   ├── workflows.R               # High-level workflows
│   ├── supervised-*.R            # Supervised wrappers (8 files)
│   ├── unsupervised-*.R          # Unsupervised wrappers (8 files)
│   ├── pipeline.R                # Advanced pipelines
│   ├── model-selection.R         # Cross-validation
│   ├── tuning.R                  # Hyperparameter tuning
│   ├── interactions.R            # Interaction effects
│   ├── diagnostics.R             # Model diagnostics
│   ├── metrics.R                 # Evaluation metrics
│   ├── visualization.R           # Unified plotting
│   └── tables.R                  # Formatted gt tables
├── man/                          # Documentation (auto-generated)
│   └── figures/
│       └── logo.png              # Hex sticker
├── tests/
│   └── testthat/
└── vignettes/
```

## Dependencies

### Core Dependencies (Imports)

- **Tidyverse**: dplyr, ggplot2, tibble, tidyr, purrr, rlang, magrittr
- **Supervised ML**: glmnet, randomForest, rpart, gbm, e1071, nnet
- **Unsupervised ML**: cluster, dbscan, MASS, smacof
- **Evaluation**: ROCR, yardstick, rsample

### Optional Dependencies (Suggests)

- **Data ingestion**: readr, readxl, nanoparquet, jsonlite, DBI, RSQLite,
  RPostgres, RMariaDB, bigrquery, paws.storage
- **Tables**: gt
- **Deep learning**: keras, tensorflow
- **Gradient boosting**: xgboost
- **Market basket**: arules, arulesViz
- **Dashboards**: shiny, shinydashboard
- Various visualization packages

## Acknowledgments

tidylearn wraps the excellent work of many R package authors. The algorithms are
implemented in stats, glmnet, randomForest, xgboost, gbm, e1071, nnet, rpart,
cluster, dbscan, MASS, smacof, and keras/tensorflow.
