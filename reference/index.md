# Package index

## Reading data

One interface over files, databases and cloud sources. Every reader
returns a tibble.

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  : Read data from diverse sources
- [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md)
  : Read from Google BigQuery
- [`tl_read_csv()`](https://tidylearn.sheetsolved.com/reference/tl_read_csv.md)
  : Read a CSV file
- [`tl_read_db()`](https://tidylearn.sheetsolved.com/reference/tl_read_db.md)
  : Read from a DBI database connection
- [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)
  : Read all matching files from a directory
- [`tl_read_excel()`](https://tidylearn.sheetsolved.com/reference/tl_read_excel.md)
  : Read an Excel file
- [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
  : Read from GitHub
- [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md)
  : Read a JSON file
- [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  : Read from Kaggle
- [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)
  : Read from a MySQL/MariaDB database
- [`tl_read_parquet()`](https://tidylearn.sheetsolved.com/reference/tl_read_parquet.md)
  : Read a Parquet file
- [`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md)
  : Read from a PostgreSQL database
- [`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md)
  : Read an RData file
- [`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md)
  : Read an RDS file
- [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  : Read from Amazon S3
- [`tl_read_sqlite()`](https://tidylearn.sheetsolved.com/reference/tl_read_sqlite.md)
  : Read from a SQLite database
- [`tl_read_tsv()`](https://tidylearn.sheetsolved.com/reference/tl_read_tsv.md)
  : Read a TSV file
- [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
  : Read data from a zip archive

## Modelling

Fit, split, evaluate and cross-validate. Reach the underlying model
object through the `$fit` slot.

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  : Create a tidylearn model
- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  : Split data into train and test sets
- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  : Data Preprocessing for tidylearn
- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  : Evaluate a tidylearn model
- [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
  : Model coefficients as a tibble
- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) :
  Cross-validation for tidylearn models
- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  : Compare models using cross-validation
- [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  : Calculate classification metrics
- [`tl_get_best_model()`](https://tidylearn.sheetsolved.com/reference/tl_get_best_model.md)
  : Get the best model from a pipeline
- [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  : Stratified Features via Clustering
- [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  : Semi-Supervised Learning via Clustering
- [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)
  : Anomaly-Aware Supervised Learning
- [`tl_step_selection()`](https://tidylearn.sheetsolved.com/reference/tl_step_selection.md)
  : Perform stepwise selection on a linear model
- [`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)
  : Integration Functions: Combining Supervised and Unsupervised
  Learning
- [`tl_add_cluster_features()`](https://tidylearn.sheetsolved.com/reference/tl_add_cluster_features.md)
  : Cluster-Based Features

## Pipelines

Compose the steps above into something you can save and re-run.

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  : Create a modeling pipeline
- [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  : Run a tidylearn pipeline
- [`tl_save_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_save_pipeline.md)
  : Save a pipeline to disk
- [`tl_load_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_load_pipeline.md)
  : Load a pipeline from disk
- [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  : Make predictions using a pipeline
- [`tl_compare_pipeline_models()`](https://tidylearn.sheetsolved.com/reference/tl_compare_pipeline_models.md)
  : Compare models from a pipeline

## Tuning and AutoML

- [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  : Tune a deep learning model
- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  : Tune hyperparameters for a model using grid search
- [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  : Tune a neural network model
- [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  : Tune hyperparameters using random search
- [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  : Tune XGBoost hyperparameters
- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  : Auto ML: Automated Machine Learning Workflow
- [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
  : Find important interactions automatically
- [`tl_default_param_grid()`](https://tidylearn.sheetsolved.com/reference/tl_default_param_grid.md)
  : Create pre-defined parameter grids for common models

## Diagnostics

What the model got wrong, and which observations drove it.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  : Check model assumptions
- [`tl_diagnostic_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_diagnostic_dashboard.md)
  : Create a comprehensive diagnostic dashboard
- [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md)
  : Calculate influence measures for a linear model
- [`tl_detect_outliers()`](https://tidylearn.sheetsolved.com/reference/tl_detect_outliers.md)
  : Detect outliers in the data
- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  : Calculate partial effects based on a model with interactions
- [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md)
  : Test for significant interactions between variables
- [`tl_test_model_difference()`](https://tidylearn.sheetsolved.com/reference/tl_test_model_difference.md)
  : Perform statistical comparison of models using cross-validation
- [`tl_explore()`](https://tidylearn.sheetsolved.com/reference/tl_explore.md)
  : Exploratory Data Analysis Workflow
- [`tl_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_dashboard.md)
  : Create interactive visualization dashboard for a model

## Unsupervised learning

Clustering, ordination and dimension reduction, each returning a tibble
rather than a fitted object you have to take apart.

- [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md)
  : Tidy Apriori Algorithm
- [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  : Tidy CLARA (Clustering Large Applications)
- [`tidy_cutree()`](https://tidylearn.sheetsolved.com/reference/tidy_cutree.md)
  : Cut Hierarchical Clustering Tree
- [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  : Tidy DBSCAN Clustering
- [`tidy_dendrogram()`](https://tidylearn.sheetsolved.com/reference/tidy_dendrogram.md)
  : Plot Dendrogram
- [`tidy_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_dist.md)
  : Tidy Distance Matrix Computation
- [`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)
  : Tidy Gap Statistic
- [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  : Gower Distance Calculation
- [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)
  : Tidy Hierarchical Clustering
- [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md)
  : Tidy K-Means Clustering
- [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md)
  : Compute k-NN Distances
- [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)
  : Tidy Multidimensional Scaling
- [`tidy_mds_classical()`](https://tidylearn.sheetsolved.com/reference/tidy_mds_classical.md)
  : Classical (Metric) MDS
- [`tidy_mds_kruskal()`](https://tidylearn.sheetsolved.com/reference/tidy_mds_kruskal.md)
  : Kruskal's Non-metric MDS
- [`tidy_mds_sammon()`](https://tidylearn.sheetsolved.com/reference/tidy_mds_sammon.md)
  : Sammon Mapping
- [`tidy_mds_smacof()`](https://tidylearn.sheetsolved.com/reference/tidy_mds_smacof.md)
  : SMACOF MDS (Metric or Non-metric)
- [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  : Tidy PAM (Partitioning Around Medoids)
- [`tidy_pca()`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md)
  : Tidy Principal Component Analysis
- [`tidy_pca_biplot()`](https://tidylearn.sheetsolved.com/reference/tidy_pca_biplot.md)
  : Create PCA Biplot
- [`tidy_pca_screeplot()`](https://tidylearn.sheetsolved.com/reference/tidy_pca_screeplot.md)
  : Create PCA Scree Plot
- [`tidy_rules()`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md)
  : Convert Association Rules to Tidy Tibble
- [`tidy_silhouette()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette.md)
  : Tidy Silhouette Analysis
- [`tidy_silhouette_analysis()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette_analysis.md)
  : Silhouette Analysis Across Multiple k Values
- [`augment_dbscan()`](https://tidylearn.sheetsolved.com/reference/augment_dbscan.md)
  : Augment Data with DBSCAN Cluster Assignments
- [`augment_hclust()`](https://tidylearn.sheetsolved.com/reference/augment_hclust.md)
  : Augment Data with Hierarchical Cluster Assignments
- [`augment_kmeans()`](https://tidylearn.sheetsolved.com/reference/augment_kmeans.md)
  : Augment Data with K-Means Cluster Assignments
- [`augment_pam()`](https://tidylearn.sheetsolved.com/reference/augment_pam.md)
  : Augment Data with PAM Cluster Assignments
- [`augment_pca()`](https://tidylearn.sheetsolved.com/reference/augment_pca.md)
  : Augment Original Data with PCA Scores
- [`optimal_clusters()`](https://tidylearn.sheetsolved.com/reference/optimal_clusters.md)
  : Find Optimal Number of Clusters
- [`optimal_hclust_k()`](https://tidylearn.sheetsolved.com/reference/optimal_hclust_k.md)
  : Determine Optimal Number of Clusters for Hierarchical Clustering
- [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  : Compare Multiple Clustering Results
- [`compare_distances()`](https://tidylearn.sheetsolved.com/reference/compare_distances.md)
  : Compare Distance Methods
- [`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)
  : Calculate Cluster Validation Metrics
- [`calc_wss()`](https://tidylearn.sheetsolved.com/reference/calc_wss.md)
  : Calculate Within-Cluster Sum of Squares for Different k
- [`explore_dbscan_params()`](https://tidylearn.sheetsolved.com/reference/explore_dbscan_params.md)
  : Explore DBSCAN Parameters
- [`get_pca_loadings()`](https://tidylearn.sheetsolved.com/reference/get_pca_loadings.md)
  : Get PCA Loadings in Wide Format
- [`get_pca_variance()`](https://tidylearn.sheetsolved.com/reference/get_pca_variance.md)
  : Get Variance Explained Summary
- [`suggest_eps()`](https://tidylearn.sheetsolved.com/reference/suggest_eps.md)
  : Suggest eps Parameter for DBSCAN
- [`standardize_data()`](https://tidylearn.sheetsolved.com/reference/standardize_data.md)
  : Standardize Data
- [`create_cluster_dashboard()`](https://tidylearn.sheetsolved.com/reference/create_cluster_dashboard.md)
  : Create Summary Dashboard

## Association rules

- [`inspect_rules()`](https://tidylearn.sheetsolved.com/reference/inspect_rules.md)
  : Inspect Association Rules
- [`filter_rules_by_item()`](https://tidylearn.sheetsolved.com/reference/filter_rules_by_item.md)
  : Filter Rules by Item
- [`find_related_items()`](https://tidylearn.sheetsolved.com/reference/find_related_items.md)
  : Find Related Items
- [`recommend_products()`](https://tidylearn.sheetsolved.com/reference/recommend_products.md)
  : Generate Product Recommendations
- [`summarize_rules()`](https://tidylearn.sheetsolved.com/reference/summarize_rules.md)
  : Summarize Association Rules
- [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md)
  : Visualize Association Rules

## Plots

ggplot2 objects, returned rather than printed, so you can keep editing
them.

- [`tl_plot_cv_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_cv_comparison.md)
  : Plot comparison of cross-validation results
- [`tl_plot_cv_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_cv_results.md)
  : Plot cross-validation results
- [`tl_plot_deep_architecture()`](https://tidylearn.sheetsolved.com/reference/tl_plot_deep_architecture.md)
  : Plot deep learning model architecture
- [`tl_plot_deep_history()`](https://tidylearn.sheetsolved.com/reference/tl_plot_deep_history.md)
  : Plot deep learning model training history
- [`tl_plot_gain()`](https://tidylearn.sheetsolved.com/reference/tl_plot_gain.md)
  : Plot gain chart for a classification model
- [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  : Plot feature importance across multiple models
- [`tl_plot_importance_regularized()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_regularized.md)
  : Plot variable importance for a regularized model
- [`tl_plot_influence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_influence.md)
  : Plot influence diagnostics
- [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  : Plot interaction effects
- [`tl_plot_intervals()`](https://tidylearn.sheetsolved.com/reference/tl_plot_intervals.md)
  : Create confidence and prediction interval plots
- [`tl_plot_lift()`](https://tidylearn.sheetsolved.com/reference/tl_plot_lift.md)
  : Plot lift chart for a classification model
- [`tl_plot_model_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_model_comparison.md)
  : Plot model comparison
- [`tl_plot_nn_architecture()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_architecture.md)
  : Plot neural network architecture
- [`tl_plot_nn_tuning()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_tuning.md)
  : Plot a neural network tuning grid
- [`tl_plot_partial_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_partial_dependence.md)
  : Plot partial dependence for tree-based models
- [`tl_plot_regularization_cv()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_cv.md)
  : Plot cross-validation results for a regularized model
- [`tl_plot_regularization_path()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_path.md)
  : Plot regularization path for a regularized model
- [`tl_plot_svm_boundary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_svm_boundary.md)
  : Plot SVM decision boundary
- [`tl_plot_svm_tuning()`](https://tidylearn.sheetsolved.com/reference/tl_plot_svm_tuning.md)
  : Plot SVM tuning results
- [`tl_plot_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tree.md)
  : Plot a decision tree
- [`tl_plot_tuning_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tuning_results.md)
  : Plot hyperparameter tuning results
- [`tl_plot_xgboost_importance()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_importance.md)
  : Plot feature importance for an XGBoost model
- [`tl_plot_xgboost_shap_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_dependence.md)
  : Plot SHAP dependence for a specific feature
- [`tl_plot_xgboost_shap_summary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_summary.md)
  : Plot SHAP summary for XGBoost model
- [`tl_plot_xgboost_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_tree.md)
  : Plot XGBoost tree visualization
- [`plot_cluster_comparison()`](https://tidylearn.sheetsolved.com/reference/plot_cluster_comparison.md)
  : Create Cluster Comparison Plot
- [`plot_cluster_sizes()`](https://tidylearn.sheetsolved.com/reference/plot_cluster_sizes.md)
  : Plot Cluster Size Distribution
- [`plot_clusters()`](https://tidylearn.sheetsolved.com/reference/plot_clusters.md)
  : Plot Clusters in 2D Space
- [`plot_dendrogram()`](https://tidylearn.sheetsolved.com/reference/plot_dendrogram.md)
  : Plot Dendrogram with Cluster Highlights
- [`plot_distance_heatmap()`](https://tidylearn.sheetsolved.com/reference/plot_distance_heatmap.md)
  : Create Distance Heatmap
- [`plot_elbow()`](https://tidylearn.sheetsolved.com/reference/plot_elbow.md)
  : Create Elbow Plot for K-Means
- [`plot_gap_stat()`](https://tidylearn.sheetsolved.com/reference/plot_gap_stat.md)
  : Plot Gap Statistic
- [`plot_knn_dist()`](https://tidylearn.sheetsolved.com/reference/plot_knn_dist.md)
  : Plot k-NN Distance Plot
- [`plot_mds()`](https://tidylearn.sheetsolved.com/reference/plot_mds.md)
  : Plot MDS Configuration
- [`plot_silhouette()`](https://tidylearn.sheetsolved.com/reference/plot_silhouette.md)
  : Plot Silhouette Analysis
- [`plot_variance_explained()`](https://tidylearn.sheetsolved.com/reference/plot_variance_explained.md)
  : Plot Variance Explained (PCA)
- [`tl_xgboost_shap()`](https://tidylearn.sheetsolved.com/reference/tl_xgboost_shap.md)
  : Generate SHAP values for XGBoost model interpretation

## Tables

Formatted gt tables for reporting.

- [`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
  : Create formatted tables for tidylearn models
- [`tl_table_clusters()`](https://tidylearn.sheetsolved.com/reference/tl_table_clusters.md)
  : Formatted cluster summary table
- [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  : Formatted model coefficients table
- [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)
  : Compare multiple models in a formatted table
- [`tl_table_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_table_confusion.md)
  : Formatted confusion matrix table
- [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md)
  : Formatted feature importance table
- [`tl_table_loadings()`](https://tidylearn.sheetsolved.com/reference/tl_table_loadings.md)
  : Formatted PCA loadings table
- [`tl_table_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_table_metrics.md)
  : Formatted evaluation metrics table
- [`tl_table_variance()`](https://tidylearn.sheetsolved.com/reference/tl_table_variance.md)
  : Formatted PCA variance explained table

## Compute and cloud

Where the work runs, and what it is allowed to reach.

- [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  : Detect local GPU availability for tidylearn methods
- [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  : Advise on the best compute tier for a tidylearn fit
- [`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)
  : Transfer Learning Workflow
- [`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)
  : Allow an additional host for cloud uploads in this R session
- [`tl_cloud_allowed_hosts()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allowed_hosts.md)
  : Hosts tidylearn will currently upload to
- [`tl_cloud_consent()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_consent.md)
  : Grant or revoke cloud upload consent for this R session
- [`tl_cloud_jobs()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_jobs.md)
  : Cloud jobs submitted in this R session

## Utilities

- [`tl_version()`](https://tidylearn.sheetsolved.com/reference/tl_version.md)
  : Get tidylearn version information
- [`` `%>%` ``](https://tidylearn.sheetsolved.com/reference/pipe.md) :
  Pipe operator

## Methods

What [`print()`](https://rdrr.io/r/base/print.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html),
[`predict()`](https://rdrr.io/r/stats/predict.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) do to each object the
package returns.

- [`plot(`*`<tidylearn_eda>`*`)`](https://tidylearn.sheetsolved.com/reference/plot.tidylearn_eda.md)
  : Plot EDA results

- [`plot(`*`<tidylearn_model>`*`)`](https://tidylearn.sheetsolved.com/reference/plot.tidylearn_model.md)
  : Plot method for tidylearn models

- [`predict(`*`<tidylearn_model>`*`)`](https://tidylearn.sheetsolved.com/reference/predict.tidylearn_model.md)
  : Predict using a tidylearn model

- [`predict(`*`<tidylearn_stratified>`*`)`](https://tidylearn.sheetsolved.com/reference/predict.tidylearn_stratified.md)
  : Predict from stratified models

- [`predict(`*`<tidylearn_transfer>`*`)`](https://tidylearn.sheetsolved.com/reference/predict.tidylearn_transfer.md)
  : Predict with transfer learning model

- [`print(`*`<tidy_apriori>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_apriori.md)
  : Print Method for tidy_apriori

- [`print(`*`<tidy_dbscan>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_dbscan.md)
  : Print Method for tidy_dbscan

- [`print(`*`<tidy_gap>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_gap.md)
  : Print Method for tidy_gap

- [`print(`*`<tidy_hclust>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_hclust.md)
  : Print Method for tidy_hclust

- [`print(`*`<tidy_kmeans>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_kmeans.md)
  : Print Method for tidy_kmeans

- [`print(`*`<tidy_mds>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_mds.md)
  : Print Method for tidy_mds

- [`print(`*`<tidy_pam>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_pam.md)
  : Print Method for tidy_pam

- [`print(`*`<tidy_pca>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_pca.md)
  : Print Method for tidy_pca

- [`print(`*`<tidy_silhouette>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidy_silhouette.md)
  : Print Method for tidy_silhouette

- [`print(`*`<tidylearn_automl>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_automl.md)
  : Print auto ML results

- [`print(`*`<tidylearn_compute_advice>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_compute_advice.md)
  :

  Print method for `tidylearn_compute_advice` objects

- [`print(`*`<tidylearn_data>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_data.md)
  : Print a tidylearn_data object

- [`print(`*`<tidylearn_eda>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_eda.md)
  : Print EDA results

- [`print(`*`<tidylearn_gpu_check>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_gpu_check.md)
  :

  Print method for `tidylearn_gpu_check` objects

- [`print(`*`<tidylearn_model>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_model.md)
  : Print method for tidylearn models

- [`print(`*`<tidylearn_pipeline>`*`)`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_pipeline.md)
  : Print a tidylearn pipeline

- [`summary(`*`<tidylearn_model>`*`)`](https://tidylearn.sheetsolved.com/reference/summary.tidylearn_model.md)
  : Summary method for tidylearn models

- [`summary(`*`<tidylearn_pipeline>`*`)`](https://tidylearn.sheetsolved.com/reference/summary.tidylearn_pipeline.md)
  : Summarize a tidylearn pipeline

## Concepts

Topic pages covering a whole area rather than a single function. Start
here when you want the shape of something before the arguments.

- [`tidylearn-classification`](https://tidylearn.sheetsolved.com/reference/tidylearn-classification.md)
  : Classification Functions for tidylearn
- [`tidylearn-cloud-consent`](https://tidylearn.sheetsolved.com/reference/tidylearn-cloud-consent.md)
  : Cloud data-egress consent for tidylearn
- [`tidylearn-cloud-cost`](https://tidylearn.sheetsolved.com/reference/tidylearn-cloud-cost.md)
  : Cost controls for tidylearn cloud compute
- [`tidylearn-cloud-endpoint`](https://tidylearn.sheetsolved.com/reference/tidylearn-cloud-endpoint.md)
  : Cloud endpoint resolution for tidylearn
- [`tidylearn-cloud-serialize`](https://tidylearn.sheetsolved.com/reference/tidylearn-cloud-serialize.md)
  : Model serialisation for tidylearn cloud compute
- [`tidylearn-coefficients`](https://tidylearn.sheetsolved.com/reference/tidylearn-coefficients.md)
  : Coefficient Inference for tidylearn
- [`tidylearn-core`](https://tidylearn.sheetsolved.com/reference/tidylearn-core.md)
  : tidylearn: A Unified Tidy Interface to R's Machine Learning
  Ecosystem
- [`tidylearn-deep-learning`](https://tidylearn.sheetsolved.com/reference/tidylearn-deep-learning.md)
  : Deep Learning for tidylearn
- [`tidylearn-diagnostics`](https://tidylearn.sheetsolved.com/reference/tidylearn-diagnostics.md)
  : Advanced Diagnostics Functions for tidylearn
- [`tidylearn-interactions`](https://tidylearn.sheetsolved.com/reference/tidylearn-interactions.md)
  : Interaction Analysis Functions for tidylearn
- [`tidylearn-metrics`](https://tidylearn.sheetsolved.com/reference/tidylearn-metrics.md)
  : Metrics Functionality for tidylearn
- [`tidylearn-model-selection`](https://tidylearn.sheetsolved.com/reference/tidylearn-model-selection.md)
  : Model Selection Functions for tidylearn
- [`tidylearn-neural-networks`](https://tidylearn.sheetsolved.com/reference/tidylearn-neural-networks.md)
  : Neural Networks for tidylearn
- [`tidylearn-pipeline`](https://tidylearn.sheetsolved.com/reference/tidylearn-pipeline.md)
  : Model Pipeline Functions for tidylearn
- [`tidylearn-read-backends`](https://tidylearn.sheetsolved.com/reference/tidylearn-read-backends.md)
  : Data Reading Backends for tidylearn
- [`tidylearn-read`](https://tidylearn.sheetsolved.com/reference/tidylearn-read.md)
  : Data Reading Functions for tidylearn
- [`tidylearn-regression`](https://tidylearn.sheetsolved.com/reference/tidylearn-regression.md)
  : Regression Functions for tidylearn
- [`tidylearn-regularization`](https://tidylearn.sheetsolved.com/reference/tidylearn-regularization.md)
  : Regularization Functions for tidylearn
- [`tidylearn-svm`](https://tidylearn.sheetsolved.com/reference/tidylearn-svm.md)
  : Support Vector Machines for tidylearn
- [`tidylearn-tables`](https://tidylearn.sheetsolved.com/reference/tidylearn-tables.md)
  : Table Functions for tidylearn
- [`tidylearn-trees`](https://tidylearn.sheetsolved.com/reference/tidylearn-trees.md)
  : Tree-based Methods for tidylearn
- [`tidylearn-tuning`](https://tidylearn.sheetsolved.com/reference/tidylearn-tuning.md)
  : Hyperparameter Tuning Functions for tidylearn
- [`tidylearn-visualization`](https://tidylearn.sheetsolved.com/reference/tidylearn-visualization.md)
  : Visualization Functions for tidylearn
- [`tidylearn-workflows`](https://tidylearn.sheetsolved.com/reference/tidylearn-workflows.md)
  : High-Level Workflows for Common Machine Learning Patterns
- [`tidylearn-xgboost`](https://tidylearn.sheetsolved.com/reference/tidylearn-xgboost.md)
  : XGBoost Functions for tidylearn
