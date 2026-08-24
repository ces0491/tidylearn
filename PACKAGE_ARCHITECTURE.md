# tidylearn Package Architecture

This document is the map of the source tree: which file holds what, and
why the modules are cut where they are. What the package is for is in
the README’s
[Overview](https://tidylearn.sheetsolved.com/README.html#overview), and
the reasoning behind the design is in its
[Philosophy](https://tidylearn.sheetsolved.com/README.html#philosophy) —
restating either here would give the project two versions to keep in
step.

tidylearn implements no algorithms. Each module wraps an established
package, so the module boundaries follow the packages rather than the
statistics — which is why `supervised-trees.R` covers rpart,
randomForest and gbm together, and why regularisation gets a file of its
own.

## Wrapped Packages

The method-to-package mapping is maintained in one place, the
[README](https://tidylearn.sheetsolved.com/README.html#wrapped-packages),
and repeated in
[`vignette("getting-started")`](https://tidylearn.sheetsolved.com/articles/getting-started.md).
Three copies of the same table drift; this document links to it rather
than adding a fourth.

[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
dispatches on 20 methods: 13 supervised and 7 unsupervised. The
authoritative list is `supervised_methods` and `unsupervised_methods` in
`R/core.R`, which is what the error message for an unknown method
prints.

## Package Structure

### Data Ingestion Module

#### `read.R` - Unified Data Reading

- **[`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)**:
  Single entry point that auto-detects format and dispatches
- **[`tl_read_csv()`](https://tidylearn.sheetsolved.com/reference/tl_read_csv.md)**,
  **[`tl_read_tsv()`](https://tidylearn.sheetsolved.com/reference/tl_read_tsv.md)**:
  Delimited files via readr (base R fallback)
- **[`tl_read_excel()`](https://tidylearn.sheetsolved.com/reference/tl_read_excel.md)**:
  Excel files via readxl
- **[`tl_read_parquet()`](https://tidylearn.sheetsolved.com/reference/tl_read_parquet.md)**:
  Parquet files via nanoparquet
- **[`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md)**:
  JSON files via jsonlite
- **[`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md)**,
  **[`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md)**:
  Native R formats via base R
- **[`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)**:
  Scan directories for data files
- **[`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)**:
  Extract and read from zip archives
- Multi-path support: `tl_read(c("a.csv", "b.csv"))` row-binds with
  `source_file` column
- All readers return `tidylearn_data` objects (tibble subclass with
  source metadata)

#### `read-backends.R` - Database and Cloud Readers

- **[`tl_read_db()`](https://tidylearn.sheetsolved.com/reference/tl_read_db.md)**:
  Query any live DBI connection
- **[`tl_read_sqlite()`](https://tidylearn.sheetsolved.com/reference/tl_read_sqlite.md)**:
  Auto-connect to SQLite files via RSQLite
- **[`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md)**:
  PostgreSQL via RPostgres
- **[`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)**:
  MySQL/MariaDB via RMariaDB
- **[`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md)**:
  Google BigQuery via bigrquery
- **[`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)**:
  Amazon S3 via paws.storage
- **[`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)**:
  Raw file download from repositories
- **[`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)**:
  Dataset download via Kaggle CLI

All backend packages are suggested dependencies, checked at call time.

### Core Modules

#### `core.R` - Unified Model Interface

- **[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)**:
  Single entry point that dispatches to underlying packages
- Automatic routing to supervised/unsupervised implementations
- Unified S3 methods: [`print()`](https://rdrr.io/r/base/print.html),
  [`summary()`](https://rdrr.io/r/base/summary.html),
  [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html)
- Raw model accessible via `$fit` slot

``` r

# Same interface, different underlying packages
supervised <- tl_model(iris, Species ~ ., method = "forest")
supervised$fit  # Access randomForest::randomForest() result

unsupervised <- tl_model(iris[,1:4], method = "kmeans", k = 3)
unsupervised$fit  # Access stats::kmeans() result
```

#### `integration.R` - Workflow Orchestration

Functions that coordinate multiple wrapped packages:

- **[`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)**:
  Use PCA/MDS as preprocessing for supervised learning
- **[`tl_add_cluster_features()`](https://tidylearn.sheetsolved.com/reference/tl_add_cluster_features.md)**:
  Add cluster membership as features
- **[`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)**:
  Cluster-based label propagation
- **[`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)**:
  Outlier detection before supervised modeling
- **[`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)**:
  Cluster-specific supervised models

#### `workflows.R` - High-Level Workflows

- **[`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)**:
  Automated ML that tries multiple approaches
- **[`tl_explore()`](https://tidylearn.sheetsolved.com/reference/tl_explore.md)**:
  Unsupervised EDA — PCA, cluster count, clustering and a distance
  summary in one call
- **[`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)**:
  Unsupervised pre-training with supervised fine-tuning

#### `preprocessing.R` - Unified Data Preparation

- **[`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)**:
  Imputation, scaling and encoding, with the parameters kept so the
  transformation can be replayed
- **[`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)**:
  Train-test splitting with stratification

### Supervised Learning Modules

Each module wraps a specific package:

- `supervised-classification.R`: Wraps stats::glm for logistic
  regression
- `supervised-regression.R`: Wraps stats::lm for linear/polynomial
  regression
- `supervised-regularization.R`: Wraps glmnet for ridge/LASSO/elastic
  net
- `supervised-trees.R`: Wraps rpart, randomForest, gbm
- `supervised-svm.R`: Wraps e1071::svm
- `supervised-neural-networks.R`: Wraps nnet
- `supervised-deep-learning.R`: Wraps keras (optional)
- `supervised-xgboost.R`: Wraps xgboost (optional)

### Unsupervised Learning Modules

Each module wraps specific packages:

- `unsupervised-pca.R`: Wraps stats::prcomp
- `unsupervised-mds.R`: Wraps stats::cmdscale, MASS::isoMDS, smacof
- `unsupervised-clustering.R`: Wraps stats::kmeans, cluster::pam,
  cluster::clara
- `unsupervised-hclust.R`: Wraps stats::hclust
- `unsupervised-dbscan.R`: Wraps dbscan::dbscan
- `unsupervised-market-basket.R`: Wraps arules (optional)
- `unsupervised-distance.R`: Distance metric utilities
- `unsupervised-validation.R`: Cluster validation metrics

### Publishing Modules

- `visualization.R`: Unified ggplot2-based plotting —
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) dispatches by
  model type, plus specialized plot functions (`tl_plot_*()`)
- `tables.R`: Formatted `gt` tables —
  [`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
  dispatches by model type, plus
  [`tl_table_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_table_metrics.md),
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md),
  [`tl_table_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_table_confusion.md),
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  [`tl_table_variance()`](https://tidylearn.sheetsolved.com/reference/tl_table_variance.md),
  [`tl_table_loadings()`](https://tidylearn.sheetsolved.com/reference/tl_table_loadings.md),
  [`tl_table_clusters()`](https://tidylearn.sheetsolved.com/reference/tl_table_clusters.md),
  [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)

### Compute Backend Modules

Where a fit runs. Most methods are CPU-only and never touch these; the
modules exist so that the two methods with an upstream GPU path, and any
method too large for the local machine, have a documented route.

- `compute-detection.R`:
  [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  — parses `nvidia-smi` and checks which GPU-capable backends are
  installed, without loading Python or fitting anything
- `compute-advisor.R`:
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  — estimates runtime, peak RAM and cost across local CPU, local GPU and
  cloud tiers. Holds the method profile table and the Modal instance
  tier table
- `compute-routing.R`: `tl_resolve_compute()` — validates the `compute`
  argument, consults the advisor for `"auto"`, and applies the fallback
  rules

### Cloud Modules

Sequenced deliberately: the guards land before the code that could
transmit anything. The submission path is not implemented yet, so
`compute = "cloud"` still errors.

- `cloud-consent.R`:
  [`tl_cloud_consent()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_consent.md)
  — the data-egress consent gate. Session state lives in a namespace
  environment, never an option or a file, so it cannot outlive the
  session or be set from outside it
- `cloud-endpoint.R`: endpoint resolution and the host allowlist —
  [`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md),
  [`tl_cloud_allowed_hosts()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allowed_hosts.md).
  The single choke point that rejects any destination that is not
  `https` on an allowed host
- `cloud-serialize.R`: converts a fitted model to bytes and back. Twelve
  methods go through base R serialisation; `"deep"` needs its keras slot
  handled separately because a keras model is a Python reference
- `cloud-cost.R`: bounds what a submission can bill. Derives the job
  timeout from the advisor estimate, refuses a fit whose worst-case cost
  exceeds the budget, builds the metadata-only pre-upload summary, and
  records every submitted job in
  [`tl_cloud_jobs()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_jobs.md)
  so nothing runs invisibly

The contract these implement is `inst/security/threat-model.md`, which
is shipped with the package.

### Supporting Modules

- `pipeline.R`: Advanced modeling pipelines
- `model-selection.R`: Cross-validation, model comparison
- `tuning.R`: Hyperparameter tuning
- `interactions.R`: Interaction effects analysis
- `diagnostics.R`: Model diagnostics
- `metrics.R`: Evaluation metrics
- `utils.R`: Helper functions

## Function Naming Convention

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  / `tl_read_*()`: Data ingestion from any source
- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md):
  Create any model (main user interface)
- `tl_*()`: High-level user-facing functions
- `tl_fit_*()`: Internal fitting functions (call underlying packages)
- `tl_predict_*()`: Internal prediction functions
- `tl_plot_*()`: Visualization functions
- [`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
  / `tl_table_*()`: Formatted gt tables

## Usage Patterns

### Pattern 1: Full Pipeline

``` r

# Ingest -> Prepare -> Model -> Evaluate -> Publish
data <- tl_read("sales.csv")
split <- tl_split(data, prop = 0.7, stratify = "target")
model <- tl_model(split$train, target ~ ., method = "forest")
tl_evaluate(model, new_data = split$test)
tl_table_metrics(model, new_data = split$test)
```

### Pattern 2: Direct Model Fitting

``` r

# Wraps randomForest::randomForest()
model <- tl_model(data, y ~ ., method = "forest")

# Access raw model for package-specific functions
randomForest::varImpPlot(model$fit)
```

### Pattern 3: Workflow Orchestration

``` r

# Coordinates multiple packages
eda <- tl_explore(data, response = "y")
reduced <- tl_reduce_dimensions(data, response = "y", method = "pca")
enriched <- tl_add_cluster_features(reduced$data, response = "y")
model <- tl_model(enriched, y ~ ., method = "forest")
```

### Pattern 4: Automated ML

``` r

# Tries multiple wrapped packages, returns best
result <- tl_auto_ml(data, y ~ .)
best_model <- result$best_model
```

## File Organization

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
    │   ├── compute-detection.R       # Local GPU detection
    │   ├── compute-advisor.R         # Runtime/RAM/cost estimates
    │   ├── compute-routing.R         # compute = cpu/gpu/auto/cloud
    │   ├── cloud-consent.R           # Data-egress consent gate
    │   ├── cloud-endpoint.R          # Endpoint + host allowlist
    │   ├── cloud-serialize.R         # Model <-> bytes for transport
    │   ├── cloud-cost.R              # Timeouts, budget ceiling, job register
    │   ├── pipeline.R                # Advanced pipelines
    │   ├── model-selection.R         # Cross-validation
    │   ├── tuning.R                  # Hyperparameter tuning
    │   ├── interactions.R            # Interaction effects
    │   ├── diagnostics.R             # Model diagnostics
    │   ├── metrics.R                 # Evaluation metrics
    │   ├── visualization.R           # Unified plotting
    │   └── tables.R                  # Formatted gt tables
    ├── inst/
    │   └── security/
    │       └── threat-model.md       # Cloud compute contract
    ├── man/                          # Documentation (auto-generated)
    │   └── figures/
    │       └── logo.png              # Hex sticker
    ├── tests/
    │   └── testthat/
    └── vignettes/

## Dependencies

### Core Dependencies (Imports)

- **Tidyverse**: dplyr, ggplot2, tibble, tidyr, purrr, rlang, magrittr
- **Supervised ML**: glmnet, randomForest, rpart, gbm, e1071, nnet
- **Unsupervised ML**: cluster, dbscan, MASS, smacof
- **Evaluation**: ROCR, yardstick, rsample

### Optional Dependencies (Suggests)

- **Data ingestion**: readr, readxl, nanoparquet, jsonlite, DBI,
  RSQLite, RPostgres, RMariaDB, bigrquery, paws.storage
- **Tables**: gt
- **Deep learning**: keras, tensorflow
- **Gradient boosting**: xgboost
- **Cloud compute**: httr2
- **Market basket**: arules, arulesViz
- **Dashboards**: shiny, shinydashboard
- Various visualization packages

## Acknowledgments

Credited in the
[README](https://tidylearn.sheetsolved.com/README.html#acknowledgments),
which names each package and what it implements.
