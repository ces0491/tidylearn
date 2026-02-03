# tidylearn 0.1.0

## Initial CRAN Release

* First release of tidylearn - a unified tidy interface to R's machine learning
  ecosystem

### Features

#### Unified Interface

* `tl_model()` - Single function to fit 20+ machine learning models
* Consistent function signatures across all methods
* Tidy tibble output for all results
* Access raw model objects via `$fit` for package-specific functionality

#### Supervised Learning Methods

* Linear regression (stats::lm)
* Polynomial regression (stats::lm with poly)
* Logistic regression (stats::glm)
* Ridge, LASSO, elastic net (glmnet)
* Decision trees (rpart)
* Random forests (randomForest)
* Gradient boosting (gbm)
* XGBoost (xgboost)
* Support vector machines (e1071)
* Neural networks (nnet)
* Deep learning (keras, optional)

#### Unsupervised Learning Methods

* Principal Component Analysis (stats::prcomp)
* Multidimensional Scaling (stats, MASS, smacof)
* K-means clustering (stats::kmeans)
* PAM clustering (cluster::pam)
* CLARA clustering (cluster::clara)
* Hierarchical clustering (stats::hclust)
* DBSCAN (dbscan)

#### Additional Features

* `tl_split()` - Train/test splitting with stratification support
* `tl_prepare_data()` - Data preprocessing (scaling, imputation, encoding)
* `tl_evaluate()` - Model evaluation with multiple metrics
* `tl_auto_ml()` - Automated machine learning
* `tl_tune()` - Hyperparameter tuning with grid and random search
* Unified ggplot2-based visualization functions
* Integration workflows combining supervised and unsupervised learning

### Wrapped Packages

tidylearn wraps established R packages including: stats, glmnet, randomForest,
xgboost, gbm, e1071, nnet, rpart, cluster, dbscan, MASS, and smacof.
