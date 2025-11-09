# tidylearn: Unified Workflow Examples
# Demonstrating the power of integrating supervised and unsupervised learning

library(tidylearn)

# ============================================================================
# Example 1: Complete Integrative Workflow
# ============================================================================

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
reduced <- tl_reduce_dimensions(iris, response = "Species",
                               method = "pca", n_components = 3)
cat("Reduced from", ncol(iris) - 1, "to", ncol(reduced$data) - 2, "features\n")

# Step 3: Add cluster features
cat("\nStep 3: Feature Engineering with Clustering\n")
data_enriched <- tl_add_cluster_features(reduced$data,
                                        response = "Species",
                                        method = "kmeans", k = 3)
cat("Added cluster feature\n")

# Step 4: Train supervised model
cat("\nStep 4: Supervised Learning\n")
model <- tl_model(data_enriched, Species ~ ., method = "forest")
print(model)


# ============================================================================
# Example 2: Semi-Supervised Learning
# ============================================================================

cat("\n\nExample 2: Semi-Supervised Learning\n")
cat("====================================\n\n")

# Simulate scenario with limited labels
set.seed(42)
labeled_indices <- sample(nrow(iris), size = 15)  # Only 10% labeled!

cat("Training with only", length(labeled_indices), "labeled observations out of", nrow(iris), "\n")

semi_model <- tl_semisupervised(iris, Species ~ .,
                               labeled_indices = labeled_indices,
                               cluster_method = "kmeans",
                               supervised_method = "logistic")

cat("Semi-supervised model trained successfully\n")


# ============================================================================
# Example 3: Auto ML
# ============================================================================

cat("\n\nExample 3: Automated Machine Learning\n")
cat("======================================\n\n")

auto_result <- tl_auto_ml(iris, Species ~ .,
                         use_reduction = TRUE,
                         use_clustering = TRUE,
                         time_budget = 120)

cat("\nLeaderboard:\n")
print(auto_result$leaderboard)

cat("\nBest model:", auto_result$leaderboard$model[1], "\n")


# ============================================================================
# Example 4: Anomaly-Aware Modeling
# ============================================================================

cat("\n\nExample 4: Anomaly-Aware Modeling\n")
cat("==================================\n\n")

# Add some synthetic outliers
iris_with_outliers <- iris
iris_with_outliers[1:5, 1:4] <- iris_with_outliers[1:5, 1:4] * 3

cat("Added 5 outliers to the dataset\n")

anomaly_model <- tl_anomaly_aware(iris_with_outliers,
                                 Species ~ .,
                                 response = "Species",
                                 anomaly_method = "dbscan",
                                 action = "flag",
                                 supervised_method = "logistic",
                                 eps = 0.5, minPts = 5)

cat("Detected", anomaly_model$anomaly_info$n_anomalies, "anomalies\n")


# ============================================================================
# Example 5: Stratified Models for Heterogeneous Data
# ============================================================================

cat("\n\nExample 5: Stratified Models\n")
cat("============================\n\n")

data(mtcars)

stratified <- tl_stratified_models(mtcars, mpg ~ .,
                                  cluster_method = "kmeans", k = 3,
                                  supervised_method = "linear")

cat("Created", length(stratified$supervised_models), "cluster-specific models\n")

# Make predictions
predictions <- predict(stratified, mtcars)
cat("Predictions completed with cluster assignments\n")


# ============================================================================
# Example 6: Transfer Learning
# ============================================================================

cat("\n\nExample 6: Transfer Learning\n")
cat("============================\n\n")

transfer_model <- tl_transfer_learning(iris, Species ~ .,
                                      pretrain_method = "pca",
                                      supervised_method = "logistic",
                                      n_components = 3)

cat("Transfer learning model with PCA pre-training created\n")


# ============================================================================
# Example 7: Comprehensive Preprocessing
# ============================================================================

cat("\n\nExample 7: Comprehensive Preprocessing\n")
cat("======================================\n\n")

# Simulate data with issues
messy_data <- iris
messy_data[sample(nrow(messy_data), 10), sample(4, 1)] <- NA  # Add missing values
messy_data$redundant_col <- messy_data$Sepal.Length + rnorm(nrow(messy_data), 0, 0.01)

cat("Original data:", ncol(messy_data) - 1, "features with missing values\n")

processed <- tl_prepare_data(messy_data, Species ~ .,
                            impute_method = "mean",
                            scale_method = "standardize",
                            remove_correlated = TRUE,
                            correlation_cutoff = 0.95)

cat("Processed data:", ncol(processed$data) - 1, "features, no missing values\n")


cat("\n\n========================================\n")
cat("All examples completed successfully!\n")
cat("========================================\n")
