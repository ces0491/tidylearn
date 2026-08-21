# tidylearn: A Unified Tidy Interface to R's Machine Learning Ecosystem

Provides a unified tidyverse-compatible interface to R's machine
learning ecosystem - from data ingestion to model publishing. The
tl_read() family reads data from files ('CSV', 'Excel', 'Parquet',
'JSON'), databases ('SQLite', 'PostgreSQL', 'MySQL', 'BigQuery'), and
cloud sources ('S3', 'GitHub', 'Kaggle'). The tl_model() function wraps
established implementations from 'glmnet', 'randomForest', 'xgboost',
'e1071', 'rpart', 'gbm', 'nnet', 'cluster', 'dbscan', and others with
consistent function signatures and tidy tibble output. Results flow into
unified 'ggplot2'-based visualization and optional formatted 'gt' tables
via the tl_table() family. The underlying algorithms are unchanged;
'tidylearn' simply makes them easier to use together. Access raw model
objects via the \$fit slot for package-specific functionality. Methods
include random forests Breiman (2001)
[doi:10.1023/A:1010933404324](https://doi.org/10.1023/A%3A1010933404324)
, LASSO regression Tibshirani (1996)
[doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
, elastic net Zou and Hastie (2005)
[doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
, support vector machines Cortes and Vapnik (1995)
[doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018) , and
gradient boosting Friedman (2001)
[doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) .

## Details

tidylearn wraps established R machine learning packages behind one
consistent interface. The main entry points are:

- [`tl_read`](https://tidylearn.sheetsolved.com/reference/tl_read.md):

  Read data from files, databases and cloud sources into a tidy tibble.

- [`tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md):

  Fit any supported supervised or unsupervised method.

- [`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md):

  Score a fitted model.

- [`tl_table`](https://tidylearn.sheetsolved.com/reference/tl_table.md):

  Render results as formatted gt tables.

- [`tl_auto_ml`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md):

  Search across methods automatically.

Every fitted model keeps the underlying package's own object in its
`$fit` slot, so package-specific functionality remains available.

See
[`vignette("getting-started", package = "tidylearn")`](https://tidylearn.sheetsolved.com/articles/getting-started.md)
for a walkthrough.

## See also

Useful links:

- <https://tidylearn.sheetsolved.com>

- <https://github.com/ces0491/tidylearn>

- Report bugs at <https://github.com/ces0491/tidylearn/issues>

## Author

**Maintainer**: Cesaire Tobias <cesaire@sheetsolved.com>
