# tidylearn: A Unified Tidy Interface to R's Machine Learning Ecosystem

Core functionality for tidylearn. This package provides a unified
tidyverse-compatible interface to established R machine learning
packages including glmnet, randomForest, xgboost, e1071, rpart, gbm,
nnet, cluster, and dbscan. The underlying algorithms are unchanged -
tidylearn wraps them with consistent function signatures, tidy tibble
output, and unified ggplot2-based visualization. Supervised models keep
the wrapped object at model\$fit; unsupervised ones put it at
model\$fit\$model, alongside the tidied components.
