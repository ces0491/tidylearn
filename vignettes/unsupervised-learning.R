## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)


## ----setup--------------------------------------------------------------------
library(tidylearn)
library(dplyr)
library(ggplot2)


## -----------------------------------------------------------------------------
# Perform PCA on iris data (excluding species)
model_pca <- tl_model(iris[, 1:4], method = "pca")
print(model_pca)


## -----------------------------------------------------------------------------
# Extract variance explained
variance_explained <- model_pca$fit$variance_explained
print(variance_explained)


## -----------------------------------------------------------------------------
# Cumulative variance explained
cumsum(variance_explained$proportion)


## -----------------------------------------------------------------------------
# Transform data to principal components
pca_scores <- predict(model_pca)
head(pca_scores)


## -----------------------------------------------------------------------------
# Visualize first two components
pca_plot_data <- pca_scores %>%
  mutate(Species = iris$Species)

ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(
    title = "PCA of Iris Dataset",
    x = paste0("PC1 (", round(variance_explained$proportion[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(variance_explained$proportion[2] * 100, 1), "%)")
  ) +
  theme_minimal()


## -----------------------------------------------------------------------------
# Examine loadings (variable contributions)
loadings <- model_pca$fit$loadings
print(loadings)


## -----------------------------------------------------------------------------
# Perform MDS
model_mds <- tl_model(iris[, 1:4], method = "mds", k = 2)
print(model_mds)


## -----------------------------------------------------------------------------
# Extract MDS coordinates
mds_points <- predict(model_mds)
head(mds_points)


## -----------------------------------------------------------------------------
# Visualize MDS
mds_plot_data <- mds_points %>%
  mutate(Species = iris$Species)

ggplot(mds_plot_data, aes(x = Dim1, y = Dim2, color = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "MDS of Iris Dataset") +
  theme_minimal()


## -----------------------------------------------------------------------------
# Perform k-means with k=3
model_kmeans <- tl_model(iris[, 1:4], method = "kmeans", k = 3)
print(model_kmeans)


## -----------------------------------------------------------------------------
# Extract cluster assignments
clusters <- model_kmeans$fit$clusters
head(clusters)


## -----------------------------------------------------------------------------
# Compare clusters with actual species
table(Cluster = clusters$cluster, Species = iris$Species)


## -----------------------------------------------------------------------------
# Visualize clusters using PCA
cluster_viz <- pca_scores %>%
  mutate(
    Cluster = as.factor(clusters$cluster),
    Species = iris$Species
  )

ggplot(cluster_viz, aes(x = PC1, y = PC2, color = Cluster, shape = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "K-means Clusters vs True Species") +
  theme_minimal()


## -----------------------------------------------------------------------------
# Access cluster centers
centers <- model_kmeans$fit$centers
print(centers)


## ----eval=FALSE---------------------------------------------------------------
# # Perform PAM clustering
# model_pam <- tl_model(iris[, 1:4], method = "pam", k = 3)
# print(model_pam)
# 
# # Extract clusters
# clusters_pam <- model_pam$fit$clusters
# table(Cluster = clusters_pam$cluster, Species = iris$Species)


## -----------------------------------------------------------------------------
# Perform hierarchical clustering
model_hclust <- tl_model(iris[, 1:4], method = "hclust")
print(model_hclust)


## -----------------------------------------------------------------------------
# Plot dendrogram
plot(model_hclust$fit$model, labels = FALSE, main = "Hierarchical Clustering of Iris")


## -----------------------------------------------------------------------------
# Cut tree to get clusters
k <- 3
clusters_hc <- cutree(model_hclust$fit$model, k = k)
table(Cluster = clusters_hc, Species = iris$Species)


## -----------------------------------------------------------------------------
# Visualize hierarchical clusters
hc_viz <- pca_scores %>%
  mutate(
    Cluster = as.factor(clusters_hc),
    Species = iris$Species
  )

ggplot(hc_viz, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Hierarchical Clustering Results") +
  theme_minimal()


## ----eval=FALSE---------------------------------------------------------------
# # Perform DBSCAN
# model_dbscan <- tl_model(iris[, 1:4], method = "dbscan", eps = 0.5, minPts = 5)
# print(model_dbscan)
# 
# # Extract clusters (0 = noise/outliers)
# clusters_dbscan <- model_dbscan$fit$clusters
# table(clusters_dbscan$cluster)
# 
# # Compare with species
# table(Cluster = clusters_dbscan$cluster, Species = iris$Species)


## ----eval=FALSE---------------------------------------------------------------
# # Create larger dataset
# large_data <- iris[rep(1:nrow(iris), 10), 1:4]
# 
# # Perform CLARA
# model_clara <- tl_model(large_data, method = "clara", k = 3, samples = 5)
# print(model_clara)
# 
# # Extract clusters
# clusters_clara <- model_clara$fit$clusters


## -----------------------------------------------------------------------------
# Try different values of k
k_values <- 2:8
within_ss <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  model <- tl_model(iris[, 1:4], method = "kmeans", k = k)
  within_ss[i] <- model$fit$model$tot.withinss
}

# Plot elbow curve
elbow_data <- data.frame(k = k_values, within_ss = within_ss)

ggplot(elbow_data, aes(x = k, y = within_ss)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    title = "Elbow Method for Optimal k",
    x = "Number of Clusters (k)",
    y = "Total Within-Cluster Sum of Squares"
  ) +
  theme_minimal()


## -----------------------------------------------------------------------------
# Train clustering model
model_train <- tl_model(iris[1:100, 1:4], method = "kmeans", k = 3)

# Predict cluster assignments for new data
new_data <- iris[101:150, 1:4]
new_clusters <- predict(model_train, new_data = new_data)

head(new_clusters)


## -----------------------------------------------------------------------------
# Train PCA model
pca_train <- tl_model(iris[1:100, 1:4], method = "pca")

# Transform new data
new_pca <- predict(pca_train, new_data = new_data)
head(new_pca)


## -----------------------------------------------------------------------------
# Reduce dimensions with PCA
pca_model <- tl_model(iris[, 1:4], method = "pca")
pca_data <- predict(pca_model)

# Select first 2 components
pca_reduced <- pca_data %>% select(PC1, PC2)

# Cluster in reduced space
kmeans_pca <- tl_model(pca_reduced, method = "kmeans", k = 3)
clusters_pca <- kmeans_pca$fit$clusters

# Visualize
viz_combined <- pca_data %>%
  mutate(
    Cluster = as.factor(clusters_pca$cluster),
    Species = iris$Species
  )

ggplot(viz_combined, aes(x = PC1, y = PC2, color = Cluster, shape = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Clustering in PCA Space") +
  theme_minimal()


## -----------------------------------------------------------------------------
# Simulate customer data
set.seed(42)
customers <- data.frame(
  age = rnorm(200, 40, 15),
  income = rnorm(200, 50000, 20000),
  spending_score = rnorm(200, 50, 25)
)

# Standardize features
customers_scaled <- scale(customers) %>% as.data.frame()

# Cluster customers
customer_segments <- tl_model(customers_scaled, method = "kmeans", k = 4)
customers$segment <- customer_segments$fit$clusters$cluster

# Visualize segments
ggplot(customers, aes(x = income, y = spending_score, color = as.factor(segment))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(
    title = "Customer Segmentation",
    color = "Segment"
  ) +
  theme_minimal()


## -----------------------------------------------------------------------------
# Use PCA for feature extraction
pca_features <- tl_model(mtcars, method = "pca")

# Keep components explaining 90% of variance
var_exp <- pca_features$fit$variance_explained
cumulative_var <- cumsum(var_exp$proportion)
n_components <- which(cumulative_var >= 0.90)[1]

cat("Components needed for 90% variance:", n_components, "\n")
cat("Original features:", ncol(mtcars), "\n")
cat("Dimension reduction:", round((1 - n_components/ncol(mtcars)) * 100, 1), "%\n")


## -----------------------------------------------------------------------------
# Complete unsupervised workflow
workflow_data <- iris[, 1:4]

# 1. Reduce dimensions
pca_final <- tl_model(workflow_data, method = "pca")

# 2. Cluster in reduced space
pca_coords <- predict(pca_final) %>% select(PC1, PC2)
clusters_final <- tl_model(pca_coords, method = "kmeans", k = 3)

# 3. Visualize
final_viz <- pca_coords %>%
  mutate(
    Cluster = as.factor(clusters_final$fit$clusters$cluster),
    Species = iris$Species
  )

ggplot(final_viz, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Complete Unsupervised Workflow") +
  theme_minimal()

