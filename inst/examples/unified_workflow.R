# tidylearn: Unified Workflow Examples
#
# Seven end-to-end workflows combining supervised and unsupervised
# learning. Run the whole file with:
#
#   source(system.file("examples", "unified_workflow.R",
#                      package = "tidylearn"))
#
# Exercised by tests/testthat/test-examples.R, so it stays runnable.

library(tidylearn)

# ====================================================
# Example 1: Complete Integrative Workflow
# ====================================================

cat("Example 1: Complete Integrative Workflow\n")
cat("=========================================\n\n")

# Load data
data(iris)

# Step 1: Exploratory analysis
cat("Step 1: Exploratory Data Analysis\n")
eda <- tl_explore(iris, response = "Species")
print(eda)

# Step 2: Dimensionality reduction
cat("\nStep 2: Dimensionality Reduction\n")
reduced <- tl_reduce_dimensions(
  iris, response = "Species",
  method = "pca", n_components = 3
)
n_reduced <- sum(grepl("^PC", names(reduced$data)))
cat("Reduced from", ncol(iris) - 1, "to",
    n_reduced, "features\n")

# Step 3: Add cluster features
cat("\nStep 3: Feature Engineering with Clustering\n")
data_enriched <- tl_add_cluster_features(
  reduced$data,
  response = "Species",
  method = "kmeans", k = 3
)
cat("Added cluster feature\n")

# Step 4: Train supervised model
cat("\nStep 4: Supervised Learning\n")
model <- tl_model(data_enriched, Species ~ .,
                  method = "forest")
print(model)


# ====================================================
# Example 2: Semi-Supervised Learning
# ====================================================

cat("\n\nExample 2: Semi-Supervised Learning\n")
cat("====================================\n\n")

# Simulate scenario with limited labels
set.seed(42)
labeled_indices <- sample(nrow(iris), size = 15)

cat(
  "Training with only", length(labeled_indices),
  "labeled observations out of", nrow(iris), "\n"
)

# iris has three species, so the supervised method has to be
# multiclass-capable -- logistic regression is binary only.
semi_model <- tl_semisupervised(
  iris, Species ~ .,
  labeled_indices = labeled_indices,
  cluster_method = "kmeans",
  supervised_method = "forest"
)

cat("Propagated labels to", nrow(iris) - length(labeled_indices),
    "unlabelled observations\n")


# ====================================================
# Example 3: Auto ML
# ====================================================

cat("\n\nExample 3: Automated Machine Learning\n")
cat("======================================\n\n")

auto_result <- tl_auto_ml(
  iris, Species ~ .,
  use_reduction = TRUE,
  use_clustering = TRUE,
  time_budget = 120
)

cat("\nLeaderboard:\n")
print(auto_result$leaderboard)

cat("\nBest model:",
    auto_result$leaderboard$model[1], "\n")


# ====================================================
# Example 4: Anomaly-Aware Modeling
# ====================================================

cat("\n\nExample 4: Anomaly-Aware Modeling\n")
cat("==================================\n\n")

# Add some synthetic outliers
iris_with_outliers <- iris
iris_with_outliers[1:5, 1:4] <-
  iris_with_outliers[1:5, 1:4] * 3

cat("Added 5 outliers to the dataset\n")

anomaly_model <- tl_anomaly_aware(
  iris_with_outliers,
  Species ~ .,
  response = "Species",
  anomaly_method = "dbscan",
  action = "flag",
  supervised_method = "forest",
  eps = 0.5, minPts = 5
)

cat("Detected",
    anomaly_model$anomaly_info$n_anomalies,
    "anomalies\n")


# ====================================================
# Example 5: Stratified Models
# ====================================================

cat("\n\nExample 5: Stratified Models\n")
cat("============================\n\n")

data(mtcars)

stratified <- tl_stratified_models(
  mtcars, mpg ~ .,
  cluster_method = "kmeans", k = 3,
  supervised_method = "linear"
)

cat(
  "Created",
  length(stratified$supervised_models),
  "cluster-specific models\n"
)

# Make predictions
predictions <- predict(stratified, mtcars)
cat("Predictions completed with cluster assignments\n")


# ====================================================
# Example 6: Transfer Learning
# ====================================================

cat("\n\nExample 6: Transfer Learning\n")
cat("============================\n\n")

transfer_model <- tl_transfer_learning(
  iris, Species ~ .,
  pretrain_method = "pca",
  supervised_method = "forest",
  n_components = 3
)

cat("Transfer learning model built on",
    transfer_model$spec$method, "over 3 principal components\n")


# ====================================================
# Example 7: Preprocessing a messy frame
# ====================================================

cat("\n\nExample 7: Preprocessing a messy frame\n")
cat("======================================\n\n")

# Simulate data with issues
set.seed(7)
messy_data <- iris
messy_data[sample(nrow(messy_data), 10),
           sample(4, 1)] <- NA
messy_data$redundant_col <-
  messy_data$Sepal.Length +
  rnorm(nrow(messy_data), 0, 0.01)

cat("Original data:", ncol(messy_data) - 1,
    "features with missing values\n")

processed <- tl_prepare_data(
  messy_data, Species ~ .,
  impute_method = "mean",
  scale_method = "standardize",
  remove_correlated = TRUE,
  correlation_cutoff = 0.95
)

cat(
  "Processed data:", ncol(processed$data) - 1,
  "features, no missing values\n"
)


cat("\n\n========================================\n")
cat("All examples completed successfully!\n")
cat("========================================\n")
