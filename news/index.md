# Changelog

## tidylearn 0.6.0

### Breaking Changes

- tidylearn requires R 4.1.0 or later. The package, its README and the
  vignettes use the native `|>` pipe, which R 3.6 and 4.0 cannot parse.
  magrittr’s `%>%` is still re-exported, so existing code that pipes
  with it after
  [`library(tidylearn)`](https://tidylearn.sheetsolved.com) keeps
  working.

- tidylearn requires ggplot2 3.4.0 or later. The plots set line widths
  with the `linewidth` aesthetic, which ggplot2 3.3 ignores with an
  unknown-parameter warning and draws at the default width. The rlang
  floor rises from 0.4.0 to 1.0.0, the version ggplot2 3.4.0 imports, so
  it changes nothing for an install that meets the ggplot2 one.

- Some calls that ran in 0.5.0 are now errors, with a message naming the
  problem: each passed an argument tidylearn ignored or returned a
  meaningless result. Bug Fixes describes all of them. Among them are:

  - an argument a method cannot use, such as `maxdeth = 1` or `offset =`
    with `method = "tree"`, a name glmnet does not take with `"ridge"`,
    `"lasso"` or `"elastic_net"`, and a misspelt argument to
    [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md);
  - an [`offset()`](https://rdrr.io/r/stats/offset.html) term with a
    method that cannot apply it at
    [`predict()`](https://rdrr.io/r/stats/predict.html), which is every
    method but `"linear"`, `"polynomial"` and `"logistic"`;
  - a metric name
    [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
    or [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
    does not compute, such as `"RMSE"`;
  - `predict(type = "prob")` or `type = "class"` on a regression model,
    and a misspelt `type`;
  - `tl_prepare_data(impute_method = "knn")`, and
    [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
    with a `prop` outside (0, 1);
  - repeated model names in
    [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
    and
    [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md);
  - `type` in
    [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md),
    and an `exclude_vars` in
    [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
    that is not a predictor;
  - a numeric response in
    [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md),
    and `tl_anomaly_aware(action = "downweight")` with `"svm"` or
    `"deep"`.

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  makes a text predictor a factor before fitting, and stores it as one
  in `$data`. `method = "forest"` coded a character column as numbers by
  the values present in whichever frame it was handed, so a row’s
  prediction depended on the rows scored with it: on a 120-row frame
  drawn after `set.seed(7)`, in which group `"c"` sits near 10, three
  `"c"` rows scored alone predicted -0.11, -0.15 and 0.45, and 9.47,
  9.26 and 9.78 inside the full frame. A forest now splits on the column
  as a category, so its fitted values change, and a text column with
  more than 53 distinct values is refused by randomForest (“Can not
  handle categorical predictors with more than 53 categories”); exclude
  identifiers with `y ~ . - id`. `method = "boost"` fits a text
  predictor, where gbm refused it (“variable 1: grp is not of type
  numeric, ordered, or factor”), and `"xgboost"` predicts a single row
  of one, where it failed with “contrasts can be applied only to factors
  with 2 or more levels”. A text column used only inside a term, such as
  `as.numeric(code)`, keeps its type.

- [`predict()`](https://rdrr.io/r/stats/predict.html) reads a supervised
  model’s categorical columns against the levels it was trained on,
  before any method sees them. New data holding only some of the
  categories works with `"forest"`, which failed with “Type of
  predictors in new data do not match that of the training data” for
  `data.frame(cyl = factor("6"), wt = 3)`. A value given as text, or as
  the number a factor was made from, reads the same. A level the model
  was not trained on is an error naming the column, the level and the
  training levels; `"boost"` used to return a prediction for it, and the
  other methods’ errors came from their backends.

- **[`predict()`](https://rdrr.io/r/stats/predict.html) on a `"ridge"`,
  `"lasso"` or `"elastic_net"` model now uses `lambda_1se`, so its
  predictions change.** It used `lambda_min`, while
  [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md),
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  and importance all report `lambda_1se` by default, so the coefficients
  on display were not the ones behind the predictions. `lambda_1se` is
  the penalty the fit itself was documented as preferring for
  generalisation. Metrics, cross-validation and tuning scores for these
  methods move with it.

- A `"ridge"`, `"lasso"` or `"elastic_net"` model is now the full-data
  fit `cv.glmnet()` makes itself, in place of a second `glmnet()` run
  along the same penalties. The two agreed only to glmnet’s convergence
  tolerance, so coefficients and predictions move slightly – by up to
  0.004 at `lambda_1se` for `mpg ~ .` on `mtcars` with `set.seed(1)` –
  and now match [`coef()`](https://rdrr.io/r/stats/coef.html) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) on the `cv.glmnet`
  object kept in `attr(model$fit, "cv_results")`. When cross-validation
  chooses the largest penalty, every slope is now exactly zero: the
  second fit left one at about -6e-17, which
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md)
  ranked at importance 100 and
  [`tl_plot_importance_regularized()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_regularized.md)
  drew as a bar of 5.9e-17 (predictor `b` of 40 rows of noise drawn
  after `set.seed(3)`). Both now stop with the message that the penalty
  dropped every predictor, as for any penalty that keeps none.

- `method = "polynomial"` keeps a numeric term that is also part of an
  interaction as it is and adds its powers as `I(x^2)` up to the degree.
  For `mpg ~ wt * hp` the fit and its predictions are those of 0.5.0,
  but the terms are now `wt`, `hp`, `I(wt^2)`, `I(hp^2)` and `wt:hp`,
  where 0.5.0 named them `poly(wt, degree = 2, raw = TRUE)1`,
  `poly(wt, degree = 2, raw = TRUE)2` and so on, so
  [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
  and
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  list other names in another order.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on an unsupervised
  model returns `.obs_id`, the row names of the data predicted on, and
  as many components as the model keeps, on both the training and the
  new-data path. A two-dimensional MDS fit from
  `tl_reduce_dimensions(n_components = 2)` predicted `.obs_id` and
  `Dim1` only, and `Dim1` and `Dim2` without `.obs_id` for tibble input.
  PCA’s training projection returned every component however many were
  kept. k-means assignments of new data had no `.obs_id`, and PCA
  projections of new data numbered the rows `"1"`, `"2"` whatever they
  were called.

- `predict(type = "prob")` on an `"xgboost"` model returns a tibble, as
  every other method does. It returned a data frame.

- `tl_split(stratify = )` cuts a numeric column with more than five
  distinct values at its quartiles. Each distinct value used to be a
  stratum, and a stratum holding a single row corrupted the split:
  [`sample()`](https://rdrr.io/r/base/sample.html) on one number draws
  from `1:n`, so that stratum drew a row from elsewhere in the data –
  sometimes one already drawn – and left its own row in test. On 50 rows
  of a continuous `y` with `seed = 1`, 0.5.0 drew 50 training rows
  holding only 22 distinct observations and put 28 in test; the same
  call now splits 38/12, each quartile at `prop`. On `mtcars` stratified
  by `mpg`, most of whose values occur once, the split with `seed = 1`
  returned 43 rows from 32, and now splits 24/8. A numeric column with
  five or fewer distinct values, such as `cyl`, is still split by value.
  An integer ID such as `1:50` has more than five, so it is cut at its
  quartiles like any other numeric column. A stratum of a single row
  goes to the training set, so a level seen once is one the model was
  trained on. When such strata together hold more than a tenth of the
  rows, as in a character ID column, they are pooled into one stratum
  and split together; for a column of 50 distinct strings that is the
  unstratified draw, 40/10. A split that leaves the test set empty,
  which only a single row now does, warns.

- `tl_model(method = "deep")` holds out a random set of rows for
  `validation_split`, recorded in `$fit$validation_rows`. keras holds
  out the last rows as given, which on `iris`, sorted by species, were
  rows 121 to 150: 30 of the 50 virginica rows, so the model trained on
  20 virginica against 50 of each other class and was validated on
  virginica alone. A row with a missing value is dropped from the
  predictors and the response together; the response kept it, so keras
  trained on pairs shifted by one row. A constant predictor is no longer
  scaled by its standard deviation of 0, which turned it into `NaN` and
  gave every row the same prediction.

- [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md),
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  and
  [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  read the task from the response, as
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  does. `is_classification` defaults to `NULL`, which treats a factor or
  character response, including one the formula computes such as
  `factor(am)`, as classification. With the old default of `FALSE`,
  `tl_tune_xgboost(iris, Species ~ .)` was tuned as a regression on the
  class codes without a message, and
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  and
  [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  failed. `is_classification = FALSE` with a factor response is an
  error, and so is `is_classification = TRUE` with a computed response
  that is not a factor, such as `I(mpg > 20)`, which
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  ran; the message points to
  [`factor()`](https://rdrr.io/r/base/factor.html) on the left-hand
  side. The response is reduced to the classes it holds, so a two-class
  subset of `iris` that still declares setosa is tuned as two classes;
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  had tuned it with `multi:softprob` and a setosa probability column.

- [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  returns the best configuration as a `tidylearn_model` in `$model`, so
  [`predict()`](https://rdrr.io/r/stats/predict.html) and the deep plots
  take it; the keras model is at `$model$fit$model`. It returned the
  internal list, and [`predict()`](https://rdrr.io/r/stats/predict.html)
  on it failed with `no applicable method`. Every configuration is
  scored on the same randomly held-out rows.

- `tl_default_param_grid("xgboost")` returns a grid: the values
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  searches by default, plus `nrounds`, which that function picks by
  early stopping and
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  has to tune. It warned “Unknown method” and returned an empty list.
  `is_classification`, which changed no grid, now changes two regression
  grids: `"svm"` also tunes `epsilon`, and the large `"forest"` grid
  uses `nodesize = c(3, 5, 10)` around randomForest’s regression default
  of 5 in place of `c(1, 3, 5)`. With the default
  `is_classification = TRUE` the grids are unchanged.

- [`tl_plot_xgboost_importance()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_importance.md)
  returns a ggplot, as documented, and ranks by the measure
  `importance_type` names. It returned `xgb.plot.importance()`’s
  data.table, drew with base graphics, and ignored `importance_type`. A
  linear booster (`booster = "gblinear"`) has only coefficients, which
  `importance_type = "weight"` draws, ranked by size; left at its
  default, `importance_type` takes them for such a model.

- [`tl_plot_partial_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_partial_dependence.md)
  on a classification model draws one line per class when there are
  three or more, and the positive class (the second level) for two,
  named on the y axis. Its data has a `class` column for the class each
  row describes. The curve was the mean probability of the second class
  alone, and the multiclass `class` column held whichever class had the
  highest mean probability, which was not the class drawn.

- `...` in
  [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  and
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  reaches the supervised model only, and the new `cluster_args` takes
  the clustering step’s settings. Both stages got `...`, so `cp = 0.001`
  failed in k-means with “unused argument”, and `nstart = 5` reached
  k-means and then broke [`glm()`](https://rdrr.io/r/stats/glm.html). A
  clustering setting now goes in `cluster_args`,
  e.g. `cluster_args = list(nstart = 5)`; one left in `...`, such as
  `nstart`, `iter.max`, `algorithm`, `hclust_method` or pam’s `variant`,
  is refused with a message saying so. `sampsize`, `keep.data` and
  `trace`, which randomForest, gbm and nnet take, still reach the
  supervised model.

- [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  with a response the formula computes, such as
  `factor(am) ~ wt + hp + qsec`, writes each propagated label to the
  column the response is computed from, as that column’s own value in a
  labelled row of the same class. The returned `$data$am` therefore
  holds 0 and 1, where it held a factor.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  holds a variable it does not vary at its most frequent value, as
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  does. It held a factor at its first level: with `carb` a factor,
  `mpg ~ wt * hp + carb` was evaluated at `carb = 1` by one function and
  `carb = 2` by the other.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  returns a factor variable in its results – `var`, `by_var` or a
  variable it holds fixed – as a factor with the model’s levels, where
  it was a character vector; the values are unchanged. That is what lets
  it work for a forest with a factor predictor, which failed with
  `Type of predictors in new data do not match that of the training data`
  because the grid held the factor as text. Both it and
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  leave out a factor level no row uses: on iris without setosa, both
  predicted at setosa, which
  [`predict()`](https://rdrr.io/r/stats/predict.html) refused as a new
  level. They also leave out a missing value of a character variable,
  which
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  failed on with `0 (non-NA) cases`. A logical variable is treated as
  two-level; both functions failed with
  `variable 'lg' was fitted with type "logical" but type "numeric" was supplied`.
  A factor value given in `at_values` or `fixed_values` takes the
  column’s levels, and one that is not a level is refused.
  `mpg ~ log(wt) * hp` no longer warns that the model has no `wt:hp`
  interaction.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  reports `fit`, `lower`, `upper` and `slope` on the response scale
  whether or not `intervals = TRUE`, and its `se` column is removed.
  With intervals a logistic model’s values were log odds (`am ~ wt * hp`
  gave `fit` from -32 to 40) and without them probabilities. A glm’s
  interval is now built on the link scale and transformed back, and a
  linear model’s is the t-based interval
  [`predict.lm()`](https://rdrr.io/r/stats/predict.lm.html) gives. For a
  glm the `se` column was on the link scale. A model with no standard
  errors, such as a tree, failed with
  `$ operator is invalid for atomic vectors` and now returns point
  estimates with a message.

- [`augment_hclust()`](https://tidylearn.sheetsolved.com/reference/augment_hclust.md)
  attaches clusters by position. It joined on
  [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html),
  which counts within each group, so on a grouped tibble of `USArrests`
  cut at `k = 3` 34 of 50 rows received another row’s cluster. Data with
  a different number of rows from the tree was padded with `NA` or cut
  short without a message; it is now an error naming both counts. The
  caller’s grouping stays on the returned data, and so do its row names,
  which the join reset to `"1"`, `"2"`, …; a `cluster` column the data
  already has is kept beside the new one under dplyr’s repaired names
  (`cluster...5` and `cluster...6` for `USArrests`), where the join gave
  `cluster.x` and `cluster.y`.

- [`tidy_rules()`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md),
  and so
  [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md)’s
  `rules_tbl`, gains the list columns `lhs_items` and `rhs_items`, each
  side’s items as a character vector, after the quality measures. The
  item-matching helpers read them; a table of rules without them is
  refused with a message saying to keep them, and a
  [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md)
  result saved without them is rebuilt from the rules it carries. An
  empty rule set now gives a zero-row tibble with the usual columns. It
  gave a tibble with no columns, so
  [`recommend_products()`](https://tidylearn.sheetsolved.com/reference/recommend_products.md)
  failed with `object 'confidence' not found`,
  [`filter_rules_by_item()`](https://tidylearn.sheetsolved.com/reference/filter_rules_by_item.md)
  and
  [`find_related_items()`](https://tidylearn.sheetsolved.com/reference/find_related_items.md)
  with `object 'lhs' not found`, and
  [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md)
  when the plot was drawn; they now return zero rows, or an empty plot.

- [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)’s
  `medoid_index` is the medoid’s integer row position. `pam()` reports
  medoids by label when the distances carry labels, so
  `tidy_pam(mtcars, k = 3)` gave `"Toyota Corona"`, `"Merc 450SE"` and
  `"Duster 360"` where the same data as a tibble gave 21, 12 and 7. Both
  now give 21, 12 and 7.

- [`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)
  and
  [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  gain an `n_noise` column, after `avg_size`, counting the points
  labelled 0, DBSCAN’s label for noise. The other clustering methods
  number their clusters from 1, so it is 0 for them. The columns after
  it move one place along.

- Counts that ran a range backwards or reached
  [`cutree()`](https://rdrr.io/r/stats/cutree.html) as a vector are
  refused with a message naming the argument.
  [`get_pca_loadings()`](https://tidylearn.sheetsolved.com/reference/get_pca_loadings.md)
  and
  [`augment_pca()`](https://tidylearn.sheetsolved.com/reference/augment_pca.md)
  with `n_components = 0` returned PC1, since `1:0` is `c(1, 0)`, and
  `get_pca_loadings(n_components = 5)` on a four-component PCA returned
  all four; `n_components` must now be between 1 and the number of
  components. `tidy_cutree(k = 2:3)` returned 100 rows for 50
  observations, and now requires a single `k` or `h`.
  [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  returned a 0 x 0 tibble for an unnamed list, or for a single vector
  passed in place of a list, and failed on an unnamed entry of a named
  one; it names such entries `clustering_1`, `clustering_2`, …, and
  refuses anything that is not a list. `max_k = 1` in
  [`optimal_hclust_k()`](https://tidylearn.sheetsolved.com/reference/optimal_hclust_k.md),
  [`tidy_silhouette_analysis()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette_analysis.md)
  and
  [`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md),
  and `max_k = 0` in
  [`calc_wss()`](https://tidylearn.sheetsolved.com/reference/calc_wss.md),
  are refused by name, where the first two failed with
  `incorrect number of dimensions`,
  [`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)
  with `K.max >= 2 is not TRUE`, and
  [`calc_wss()`](https://tidylearn.sheetsolved.com/reference/calc_wss.md)
  inside [`kmeans()`](https://rdrr.io/r/stats/kmeans.html).

- The `source_file` column names each row’s file by its path below the
  directory or archive it came from, or, for paths given to
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  directly, below the deepest folder they share. It held base names, so
  `tl_read_dir(recursive = TRUE)` labelled `2023/sales.csv` and
  `2024/sales.csv` both `sales.csv`, losing a partition key held in a
  folder name. Files in one folder keep their bare names. When one file
  already has a `source_file` column, every file’s label now goes to
  `tl_source_file`; the choice was made per file, which put the other
  files’ labels in that file’s own column. The `tl_source` attribute of
  a zip member read on its own names the member’s path within the
  archive too, as `archive.zip//2024/sales.csv` where it held
  `archive.zip//sales.csv`.

- [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md),
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  on a folder, and
  [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
  without `file` read compressed CSV and TSV files and `.ndjson` files
  along with the rest, so a folder or archive holding them next to other
  data now row-binds them too: a folder of `cars.csv` and
  `cars_backup.csv.gz` returned mtcars’s 32 rows and now returns 64.
  `format = "csv"` takes the compressed CSVs as well. `pattern` restores
  0.5.0’s selection for a folder:
  `pattern = "\\.(csv|tsv|xlsx?|xlsm|parquet|json|rds|rdata|rda)$"` in
  place of the default scan, or `pattern = "\\.csv$"` in place of
  `format = "csv"`, matching the case of the extension where 0.5.0
  ignored it. In an archive, name the member with `file`. Scans still
  skip `.txt` files, which
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  reads as CSV only when named directly.

- [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  finds a requested file the Kaggle CLI saved under its base name, so
  `file = "data/train.csv"` is read where it was reported missing.
  Without `file`, the search for a data file uses the extensions its
  readers handle, from the same table as
  [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md):
  it knew six of its own, so a dataset of `.ndjson`, `.xlsm` or
  compressed CSV files had “no data files”, and where such files sit
  beside others the newest of all of them is now read.

- `tl_read_zip(format = )` selects the members of that format from an
  archive holding a single data file, as it already did for several. It
  forced the format onto a lone member, so a zip holding only
  `data.json` read with `format = "csv"` returned 0 rows with fragments
  of the JSON as column names; it is now an error saying the archive
  holds no CSV. To force a format onto a member, name it with `file`.

- On R before 4.4.0,
  [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
  and
  [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  refuse `.rds`, `.rdata` and `.rda` files unless the new
  `trust_rds = TRUE` is given. Those versions can run code embedded in a
  crafted file as it is read (CVE-2024-27322), and these readers handed
  downloaded bytes straight to
  [`readRDS()`](https://rdrr.io/r/base/readRDS.html) and
  [`load()`](https://rdrr.io/r/base/load.html). R 4.4.0 and later read
  them as before. The help pages of
  [`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md)
  and
  [`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md)
  say to read such files only from trusted sources.

- [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  no longer reports `torch`, which is neither a tidylearn backend nor a
  suggested package, so its `backends` list has three entries. It reads
  which backends are installed from the library:
  [`requireNamespace()`](https://rdrr.io/r/base/ns-load.html) loaded
  xgboost, tensorflow, keras and reticulate to learn they were there,
  and loading keras set `TF_USE_LEGACY_KERAS = "1"` for the rest of the
  session. `nvidia-smi` is given 10 seconds to answer, after which the
  check warns and reports no GPU; a hung driver held up
  [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md),
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  and every xgboost or deep fit with `compute = "auto"` or `"gpu"` for
  as long as it hung.

- [`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)
  refuses a host name of fewer than three labels. `co.uk`, `github.io`
  and `ngrok.io` were accepted, after which any site registered under
  them, such as `attacker.co.uk`, passed as an upload destination.
  Without the Public Suffix List the rule is the label count, so it also
  refuses an apex domain such as `example.com`, and a public suffix of
  three or more labels still passes. Give the endpoint’s full host name.
  Cloud submission is not wired up, so no upload could reach such a
  host.

### New Features

- [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
  returns a model’s coefficients as a tibble, with standard errors, test
  statistics, p-values and – with `conf_int = TRUE` – a confidence
  interval. It covers `"linear"`, `"polynomial"`, `"logistic"`,
  `"ridge"`, `"lasso"` and `"elastic_net"`. Until now the only
  coefficient output was
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md),
  which needs the suggested `gt` package installed, returns a formatted
  table rather than data, and carried no interval — so getting a slope
  and its interval meant reaching into `model$fit` and calling
  [`confint()`](https://rdrr.io/r/stats/confint.html) by hand.

  `exponentiate = TRUE` reports odds ratios for a classification model.
  The standard error stays on the log-odds scale it was computed on and
  is renamed `std_error_log`, because
  [`exp()`](https://rdrr.io/r/base/Log.html) of a standard error is not
  the standard error of `exp(estimate)`.

  Intervals are Wald intervals, built from the standard errors reported
  alongside them, so the interval and the p-value in a row always agree
  about whether zero is excluded. For `"linear"` and `"polynomial"` they
  match [`confint()`](https://rdrr.io/r/stats/confint.html) on the
  underlying `lm` exactly. For `"logistic"` they use the *z* the summary
  reports rather than profiling the likelihood, which
  `confint(model$fit)` still gives you.

  Regularised methods return the estimates and the `lambda` they came
  from. `conf_int = TRUE` is an error there rather than a column of
  `NA`: glmnet reports no standard errors, and a Wald interval on a
  shrunken estimate would not cover at its stated rate. A multiclass
  regularised model has a set of coefficients per class, returned under
  a `class` column and grouped by class in the table. `exponentiate` is
  refused there: glmnet’s multinomial coefficients are not relative to a
  reference class, so their exponent is not an odds ratio.

- [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  gained `conf_int`, `level` and `exponentiate`, passed through to
  [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md).
  A call without them gives the table it gave before, apart from three
  changes described elsewhere in this section: regularised coefficients
  come from the `cv.glmnet()` fit, a term the fit could not estimate
  gets a row, and the source note counts the rows the fit used. With
  `exponentiate = TRUE` a regularised table ranks terms by the size of
  the log odds ratio: ranked by the odds ratio itself, a term the
  penalty dropped (1) sorted above a strong negative effect (0.02).

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md),
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  and
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  accept `folds = nrow(data)` and leave each row out in turn, as
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  does, and so does
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  with `evaluation$cv_folds` equal to the number of rows. rsample’s
  `vfold_cv()` refused it with “Leave-one-out cross-validation is not
  supported by this function”.

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  and
  [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md)
  read newline-delimited `.ndjson` files, through
  [`jsonlite::stream_in()`](https://jeroen.r-universe.dev/jsonlite/reference/stream_in.html),
  and
  [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md)
  reads a `.json` file holding one record per line (JSON Lines). Both
  failed with “parse error: trailing garbage”: the `.ndjson` extension
  was mapped to JSON, and a JSON Lines file was read as a single
  document. A file that parses neither way is an error naming both. CSV
  and TSV files compressed with gzip, bzip2 or xz (`data.csv.gz`) are
  recognised by the extension under the compression one.

- `tl_model(method = "mds")` reaches every MDS variant.
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)’s
  own `method` argument holds `"mds"`, so
  [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)’s
  `"metric"`, `"nonmetric"`, `"sammon"` and `"kruskal"` could not be
  asked for, and `ndim = 3` failed with
  `formal argument "ndim" matched by multiple actual arguments`. The
  variant is now chosen with `mds_method` (default `"classical"`), and
  the dimension count is `ndim` or `k`; giving both with different
  values is an error.

- `tl_model(method = "hclust")` takes the linkage as `hclust_method`
  (default `"average"`), as `tl_model(method = "mds")` takes
  `mds_method`.
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)’s
  own `method` argument holds `"hclust"`, so the linkage could not be
  chosen and every tree used average linkage.

- [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)
  takes `distance = "gower"`, computed by
  [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  over every column, and `tl_model(method = "mds", distance = "gower")`
  uses the factor columns its formula names. `distance` went straight to
  [`stats::dist()`](https://rdrr.io/r/stats/dist.html), which failed
  with `invalid distance method`. Classical MDS on undefined distances,
  a pair of rows with no variable observed in both, now stops naming the
  rows and pointing to `method = "metric"` or `"nonmetric"`, which leave
  them out; [`cmdscale()`](https://rdrr.io/r/stats/cmdscale.html) said
  only `NA values not allowed in 'd'`.

- [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md)
  no longer prints arules’ 21-line mining trace on every call. It gains
  `control`, whose `verbose` defaults to `FALSE`; the rest of the list
  goes to
  [`arules::apriori()`](https://rdrr.io/pkg/arules/man/apriori.html) as
  given, and `control = list(verbose = TRUE)` brings the trace back. Its
  new `...` reaches
  [`arules::apriori()`](https://rdrr.io/pkg/arules/man/apriori.html)
  too: `appearance`, and further mining parameters such as `smax` or
  `maxtime`.
  [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  and
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  gain `...` for the arguments of
  [`cluster::clara()`](https://rdrr.io/pkg/cluster/man/clara.html) and
  [`cluster::pam()`](https://rdrr.io/pkg/cluster/man/pam.html):
  `tidy_clara(x, k = 3, correct.d = TRUE)` was `unused argument`, and
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  can now take `nstart`, `variant` or starting `medoids`.
  `cluster.only = TRUE` for both, `diss` for
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  and `medoids.x = FALSE` for
  [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  are refused, under any abbreviation R accepts, since the result is
  built from the full fit and its medoids.

### Bug Fixes

#### Fitting and prediction

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  decides between classification and regression from the response the
  formula computes. `factor(cyl) ~ wt` with `method = "linear"` was
  fitted by [`lm()`](https://rdrr.io/r/stats/lm.html) on the factor
  codes, with only backend warnings, and is now refused as a factor with
  3 classes; with `"ridge"` it was treated as regression and failed with
  “invalid to change the storage mode of a factor”, and with `"forest"`
  it returned a classification forest described as a regression. A
  response the formula computes as text, such as
  `ifelse(mpg > 20, "hi", "lo") ~ wt + hp`, is classified as the factor
  it encodes by `"nn"`, `"forest"` and `"svm"`, and by
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md):
  nnet stopped with `NA/NaN/Inf in foreign function call (arg 2)`,
  randomForest with `non-numeric argument to binary operator`, and e1071
  with `Need numeric dependent variable for regression.` A supervised
  method given a one-sided formula, or none, says it needs a two-sided
  formula naming the response. Given `~ wt + hp`, `"forest"` fitted an
  unsupervised forest, `"xgboost"` a booster with no objective, and
  `"polynomial"` a model of `wt` on itself and `hp`; `"logistic"` took
  `wt` as the response and refused it, and the other methods failed
  inside their backends, `"linear"` with “incompatible dimensions”.

- [`predict()`](https://rdrr.io/r/stats/predict.html) passes a method
  only the columns its formula uses. A column the model never saw
  counted as a predictor wherever a method expanded a `.` formula over
  the new data: an extra, mostly-missing `notes` column turned 27 of 30
  `"svm"` predictions `NA`, and an extra column gave `"xgboost"` a
  warning about columns not in the training data.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a supervised
  model refuses new data that lacks a column the model’s predictors were
  fitted on: “New data is missing predictors used at fit time: hp”.
  [`model.frame()`](https://rdrr.io/r/stats/model.frame.html) looks a
  variable up in the data and then in the formula’s environment, so
  every method took a missing column from a same-named object in the
  caller’s session, or failed with “object ‘hp’ not found” when there
  was none:
  `predict(tl_model(mtcars, mpg ~ wt + hp, method = "linear"), nd)` on
  data without `hp` returned predictions built from a global `hp`. A
  variable the formula takes from its environment, such as `k` in
  `offset(k * disp)`, is not required.
  [`tl_xgboost_shap()`](https://tidylearn.sheetsolved.com/reference/tl_xgboost_shap.md),
  [`tl_plot_xgboost_shap_summary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_summary.md)
  and
  [`tl_plot_xgboost_shap_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_dependence.md)
  check `data` the same way, where they computed SHAP values from a
  global `hp` or failed with `object 'hp' not found`.

- `predict(type = "prob")` and `type = "class"` on a regression model
  are errors. Both returned the numeric predictions, and a regression
  model ignored a misspelt type in the same way; an unrecognised type is
  now an error for every supervised model.
  [`predict()`](https://rdrr.io/r/stats/predict.html) on zero rows
  returns zero rows with the columns and types of a non-empty
  prediction, where `"logistic"` failed with “eta must be a nonempty
  numeric vector”, the class probabilities of a multiclass `"ridge"`,
  `"lasso"` or `"elastic_net"` model with “non-conformable arrays”, and
  `"xgboost"` with an input pointer misalignment.

- A supervised model fitted on a formula that subtracts a column, such
  as `y ~ . - id`, no longer needs that column at
  [`predict()`](https://rdrr.io/r/stats/predict.html).
  [`terms()`](https://rdrr.io/r/stats/terms.html) keeps a subtracted
  column among its variables, so every method but `"polynomial"` and
  `"boost"` refused new data without it, most with “object ‘id’ not
  found” – the baselines
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  fits on such a formula among them. The model is now fitted on the
  formula written out without the column; `$spec$formula`,
  [`print()`](https://rdrr.io/r/base/print.html) and every refit keep
  the formula as written, and a missing value in the subtracted column
  no longer leaves an `"svm"` prediction `NA`. A subtracted name that is
  neither a column of the data nor, for a supervised model, a variable
  in the formula’s environment – usually a misspelling, as in
  `mpg ~ . - qsce` – is an error that names it.

- **`tl_model(subset = )` fits the selected rows, with every method, and
  stores only those in `$data`.** `"linear"`, `"logistic"` and `"nn"`
  failed on `subset` with “..1 used in an incorrect context”, `"boost"`
  with “unused argument”, and `"tree"`, `"ridge"`, `"lasso"`,
  `"elastic_net"` and `"xgboost"` fitted every row, `"xgboost"` warning
  “Passed unrecognized parameters: subset”; a regularised fit with
  `subset = 1:16` used all 32 rows. `"forest"`, `"svm"` and
  `"polynomial"` fitted the selected rows but kept every row in `$data`,
  so `predict(model)` and `tl_evaluate(model)` scored rows the fit never
  saw. The other per-row arguments, such as `weights`, are taken to the
  same rows.

- **[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  passes case `weights` to [`lm()`](https://rdrr.io/r/stats/lm.html),
  [`glm()`](https://rdrr.io/r/stats/glm.html) and `rpart()`, to gbm for
  `"boost"`, to nnet for `"nn"` and to the training `xgb.DMatrix()` for
  `"xgboost"`, so weighted `"tree"` and `"xgboost"` fits change.** For
  `"linear"`, `"logistic"`, `"boost"` and `"nn"` it failed with
  `..1 used in an incorrect context`, and `"xgboost"` warned that it did
  not recognise them and fitted without them. `"svm"` and `"deep"`
  refuse `weights`: e1071 has no case weights, so a weighted svm fit was
  identical to the unweighted one while the model recorded that weights
  had been used, and keras’s `fit()` accepted them through its `...` and
  ignored them. `class.weights` still reaches e1071.

  For `"tree"`, every extra argument went to `rpart.control()`, which
  discards what it does not recognise, so `weights`, `parms`, `cost` and
  a whole `control` list were accepted and had no effect, `maxdeth = 1`
  fitted the default tree, and `offset = mtcars$hp` fitted without an
  offset. `rpart()`’s own arguments now go to `rpart()`, and an argument
  that neither `rpart()` nor `rpart.control()` takes is refused by name,
  as `rpart()` itself refuses one. An explicit `minsplit` given with a
  `control` list brings its derived `minbucket`, unless `minbucket` is
  passed too.

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  refuses an offset for `"ridge"`, `"lasso"`, `"elastic_net"`, `"tree"`,
  `"forest"`, `"boost"`, `"svm"`, `"nn"`, `"deep"` and `"xgboost"`,
  whether as an [`offset()`](https://rdrr.io/r/stats/offset.html) term
  in the formula or as the `offset` argument, and names it. For the
  regularised methods an
  [`offset()`](https://rdrr.io/r/stats/offset.html) term was left out of
  the fit – the coefficients of `mpg ~ wt + hp + offset(disp / 100)`
  were those of the fit with no offset – and an `offset` argument was
  fitted but left [`predict()`](https://rdrr.io/r/stats/predict.html)
  failing with `No newoffset provided for prediction`; their error
  points to `"linear"` or `"logistic"` with
  [`offset()`](https://rdrr.io/r/stats/offset.html) in the formula. None
  of the other seven applies an offset at
  [`predict()`](https://rdrr.io/r/stats/predict.html): rpart and gbm fit
  an [`offset()`](https://rdrr.io/r/stats/offset.html) term as a shift
  of the response that their predictions never add back, and the other
  five leave it out of the fit. Held at the same predictors, each
  method’s prediction moved by 0 when the offset moved by 100. As an
  argument, an offset was ignored without a message by `"tree"`,
  `"forest"`, `"svm"`, `"nn"` and `"deep"`; `"xgboost"` warned that
  xgboost did not recognise it, and `"boost"` stopped with gbm’s
  `unused argument`. `"linear"` and `"logistic"` still take an
  [`offset()`](https://rdrr.io/r/stats/offset.html) term, and
  `"polynomial"`, which dropped one, now applies it. An `offset`
  argument is refused with a pointer to
  [`offset()`](https://rdrr.io/r/stats/offset.html) in the formula,
  which is the form [`predict()`](https://rdrr.io/r/stats/predict.html)
  can apply to new data.

- A classification model’s classes (`$spec$response_levels`) are those
  of the rows it is fitted on. Every method but `"tree"`, `"boost"` and
  `"xgboost"` leaves out rows with a missing value, so a class whose
  every row has one is not in the fit; in `"ridge"`, `"lasso"` and
  `"elastic_net"` it stayed a level of the fitted response. With each
  setosa `Sepal.Width` in `iris` missing, `"lasso"` failed at fit with
  “one multinomial or binomial class has 1 or 0 observations; not
  allowed”, and `"nn"` failed at
  [`predict()`](https://rdrr.io/r/stats/predict.html) with “‘names’
  attribute \[3\] must be the same length as the vector \[2\]”; both now
  fit and predict the two remaining classes, and the regularised methods
  fit `Species ~ .` as versicolor against virginica. A `"logistic"` fit
  left with a single class fitted without a message, and is now refused;
  a regularised fit left with one class stops with a message saying so.

- [`update()`](https://rdrr.io/r/stats/update.html) and
  [`step()`](https://rdrr.io/r/stats/step.html) on a model’s `$fit`
  refit on the training rows. The stored call named `data`, which both
  functions evaluate in the caller’s environment: a script that called
  its full data `data` refitted an 18-row model on all 32 rows, and with
  nothing in scope called `data` the refit failed with “‘data’ must be a
  data.frame, environment, or list”. The call now refers to the data it
  was fitted with and prints as `data = <environment>$data`; a weighted
  fit, which 0.5.0 could not make, refits with its weights. The call
  names its function with its package, as in
  [`rpart::rpart`](https://rdrr.io/pkg/rpart/man/rpart.html), so
  [`update()`](https://rdrr.io/r/stats/update.html) works in a session
  that has not attached rpart, randomForest, e1071, nnet or gbm. A saved
  model holds the training data a second time: a tree fitted on 20,000
  rows saves at 1.1 MB, against 0.6 MB in 0.5.0.

- **[`predict()`](https://rdrr.io/r/stats/predict.html) on an
  `"xgboost"`, `"deep"` or `"boost"` model computes a data-dependent
  term of the formula, such as `scale(hp)` or `poly(wt, 2)`, with the
  values the training data gave it, as
  [`predict.lm()`](https://rdrr.io/r/stats/predict.lm.html) does.** The
  term was recomputed on the rows being predicted, so a row’s prediction
  depended on the rows scored with it: for `mpg ~ scale(hp) + wt` with
  `nrounds = 20`, the fifth row of `mtcars` predicted 15.0 alone and
  18.6 inside the full frame. With `"boost"` on `mtcars` stacked twice,
  `n.trees = 50` and `set.seed(1)`, row 5 predicted 14.72 alone and
  17.28 inside the frame. A matrix-valued term such as `poly(wt, 2)`
  failed to fit with boost
  (`number of items to replace is not a multiple of replacement length`);
  it now enters as one predictor per column, as with `"forest"`. A
  transform that needs nothing from training, such as `log(hp)`, is
  fitted through gbm’s formula interface as before.

- **`tl_model(method = "boost")` reads a formula variable that is not a
  column of the data from the formula’s environment, as
  [`lm()`](https://rdrr.io/r/stats/lm.html) does, at fit and at
  [`predict()`](https://rdrr.io/r/stats/predict.html).** gbm rebuilt its
  predictors in its own environment, which reaches only the global one,
  so `mpg ~ wt + z` with `z` defined inside a function failed with
  `object 'z' not found`, and where a global `z` existed as well, gbm
  fitted and predicted on the global one while the response came from
  the data.

- **`"xgboost"`, `"deep"`,
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  and
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  fit and score the response the formula computes.** The first three
  fitted `log(mpg) ~ wt + hp` to `mpg` itself, so `"xgboost"` predicted
  21.0 for the Mazda RX4, whose log mpg is 3.04, and
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  scored its folds against the raw column.

- `method = "logistic"` fits a response the formula computes. It was
  read off the first column named, so `I(mpg > 20) ~ wt` and
  `ifelse(mpg > 20, "hi", "lo") ~ wt` were refused with
  `Logistic regression needs a two-class response, but 'mpg' is numeric with 25 distinct values`,
  and `I(am + 1) ~ wt` failed with
  `Argument mu must be a nonempty numeric vector`. Its predictions,
  probabilities and coefficients are now those of
  [`glm()`](https://rdrr.io/r/stats/glm.html) on the same formula, a
  response computed as text or as numbers other than 0 and 1 being
  fitted as the factor it encodes, and the ROC, precision-recall,
  calibration and confusion plots score the computed response too. A
  computed factor whose first level no row has is refused, with a
  pointer to [`droplevels()`](https://rdrr.io/r/base/droplevels.html):
  [`glm()`](https://rdrr.io/r/stats/glm.html) would have compared every
  row with that empty class.

- **`method = "polynomial"` adds its terms to the formula as written.**
  The formula was rebuilt from the response’s name and the term labels,
  so `log(mpg) ~ wt` was fitted as a model of `mpg`: row 1 of `mtcars`
  predicted 22.91 where `log(mpg)` is 3.04, and now predicts 3.11.
  [`offset()`](https://rdrr.io/r/stats/offset.html) terms and `- 1` were
  dropped, `poly(wt, 3)` became `poly(poly(wt, 3), ...)`, and a factor
  went into [`poly()`](https://rdrr.io/r/stats/poly.html) on its integer
  codes, so on `iris` `Sepal.Length ~ .` predicted 6.96 for row 101, or
  7.66 once [`droplevels()`](https://rdrr.io/r/base/droplevels.html) had
  renumbered the levels. Each numeric main effect is now replaced by its
  polynomial, and predictions match
  [`lm()`](https://rdrr.io/r/stats/lm.html) on the same formula written
  out by hand. A numeric term that is also part of an interaction keeps
  its own term and gains `I(x^2)` up to the degree, so the interaction
  is coded as written: `mpg ~ cyl_f * wt`, with `cyl_f` a factor,
  returned an `NA` coefficient for `cyl_f8:wt`, and now fits
  `mpg ~ cyl_f * wt + I(wt^2)`. Factors,
  [`I()`](https://rdrr.io/r/base/AsIs.html) terms and bases such as
  [`poly()`](https://rdrr.io/r/stats/poly.html) or a spline’s are left
  as written.

- **[`predict()`](https://rdrr.io/r/stats/predict.html) on a
  `method = "polynomial"` model computes a data-dependent term such as
  `scale(wt)` with the training data’s centre and scale, so its
  predictions on new rows change.** Inside the
  [`poly()`](https://rdrr.io/r/stats/poly.html) or
  [`I()`](https://rdrr.io/r/base/AsIs.html) the expansion wraps it in,
  the term was recomputed on the rows being predicted, so for
  `mpg ~ scale(wt) + hp` on `mtcars` a single row predicted `NaN`, and
  `mtcars[1:5, ]` predicted 20.48 for its second row against a fitted
  value of 21.71. Cross-validation scored each held-out fold on its own
  centre and scale:
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) on
  `mpg ~ scale(wt) + hp` at `folds = 4` after `set.seed(1)` reported a
  mean rmse of 2.43, and now reports 2.34.

- **`"ridge"`, `"lasso"` and `"elastic_net"` keep every predictor of a
  formula without an intercept.** The design matrix’s first column was
  dropped as the intercept, so `mpg ~ wt + hp + disp - 1` (or `+ 0`) was
  fitted on `hp` and `disp` alone, with no message. glmnet still fits
  its own intercept; pass `intercept = FALSE` to fit without one.

- `"ridge"`, `"lasso"` and `"elastic_net"` cross-validate a sequence of
  penalties. `lambda = c(1, 0.1)` fitted the path with no penalty
  chosen, and [`predict()`](https://rdrr.io/r/stats/predict.html)
  returned 64 rows for 32. The sequence now goes to `cv.glmnet()`, which
  chooses `lambda_min` and `lambda_1se` from it (set a seed for a
  reproducible choice), so the model predicts,
  [`tl_plot_regularization_cv()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_cv.md)
  works for it, and
  [`tl_plot_regularization_path()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_path.md)
  marks one `lambda.min` and one `lambda.1se` in place of a dashed line
  at every penalty. A single `lambda` is fitted as given, as before.

- `"ridge"`, `"lasso"` and `"elastic_net"` reject an argument that
  neither `glmnet()` nor `cv.glmnet()` takes. Both discard names they do
  not know, so a misspelt `standardise = FALSE` changed nothing, and
  `strata` was ignored the same way. A cross-validation argument such as
  `type.measure` or `foldid` given with a single `lambda`, where no
  cross-validation runs, is refused as well. `nfolds`, `family`, `x` and
  `y` are refused with a message saying what sets each: `cv_folds` sets
  the number of folds, and the response sets the family. Passed beside
  tidylearn’s own value, they failed with R’s
  `formal argument "nfolds" matched by multiple actual arguments`.
  `relax = TRUE` and `gamma` are refused too, because predictions and
  coefficients come from the unrelaxed fit; `relax = TRUE` failed with
  `no applicable method for 'family' applied to an object of class "NULL"`.

- `weights` and `foldid` work with `"ridge"`, `"lasso"` and
  `"elastic_net"` on data with missing values. Incomplete rows were
  dropped from the predictors but not from these per-row values, so one
  missing value failed with
  `number of elements in weights (32) not equal to the number of rows of x (31)`,
  or `logical subscript too long` for `foldid`. They now follow the rows
  the fit keeps, as in [`lm()`](https://rdrr.io/r/stats/lm.html).

- `"ridge"`, `"lasso"` and `"elastic_net"` given a single predictor
  column, such as `mpg ~ wt`, stop with a message saying they need two,
  in place of glmnet’s `x should be a matrix with 2 or more columns`,
  which names an argument the caller never passed.

- **[`predict()`](https://rdrr.io/r/stats/predict.html) on an `"svm"`
  model returns one row per row of `new_data`, with `NA` only where a
  predictor is missing.** e1071’s `predict.svm()` dropped every row with
  a missing value in any column of `new_data`, the response and unused
  columns included, so `Ozone ~ Temp + Wind` on `airquality` returned
  111 predictions for 153 rows, and the rows that came back no longer
  lined up with the input. A classification model scored on `iris` with
  two missing values in an unused column returned 148 predictions for
  150 rows. When no row has every predictor, every row is `NA`;
  `predict.svm()` failed with `test data does not match model !`.

- `tl_model(method = "forest")` fits formulas with a transformed term,
  such as `mpg ~ log(hp) + wt` or `mpg ~ factor(cyl) + wt`.
  randomForest’s formula interface fails on them with
  `object 'hp' not found`, so for those formulas tidylearn builds the
  predictor frame itself and rebuilds it from the raw columns at
  [`predict()`](https://rdrr.io/r/stats/predict.html), each term
  computed as in training: `scale(hp)` with the training centre and
  scale, and [`poly()`](https://rdrr.io/r/stats/poly.html), `ns()` or
  `bs()` with the training coefficients, each of their columns a
  predictor of its own. A row therefore predicts the same alone as among
  other rows. Formulas of plain columns still go through randomForest’s
  formula interface.

- Values the method wrappers set themselves give way to the caller’s,
  where naming one failed with
  `formal argument matched by multiple actual arguments`: `probability`
  and `type` for `"svm"`, `linout` for `"nn"`, `verbose` and, for
  regression, `distribution` for `"boost"`, `maxit` and `trace` in
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md),
  and `verbose` in
  [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md).
  For classification, `"boost"` refuses a `distribution` other than the
  one it sets from the response, since its predictions read only
  bernoulli and multinomial fits.

- `tl_model(method = "xgboost")` keeps a row with a missing predictor,
  which xgboost routes itself, and drops a row with a missing response.
  A missing predictor failed with
  `The length of labels must equal to the number of rows in the input data`.

- `tl_model(method = "xgboost")` puts booster settings passed through
  `...`, such as `max_leaves`, `tree_method` or `booster`, into
  `params`. xgboost 3.x moved them there itself but warned that doing so
  will become an error. A setting tidylearn also sets, such as
  `objective` or `eval_metric`, takes the caller’s value; xgboost 3.x
  had refused an `objective` passed this way. `early_stopping_rounds` is
  refused by name unless a validation set is passed as `evals`
  (`watchlist` before xgboost 3.0). It failed with xgboost’s
  `For early stopping, 'evals' must have at least one element`. A
  validation set passed under the other version’s name is renamed, in
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  too, where it reaches the final fit; xgboost 3.x renamed `watchlist`
  itself with a warning that doing so will become an error.

- `tl_model(method = "xgboost", booster = "gblinear")` leaves out the
  tree parameters tidylearn sets by default (`max_depth`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `gamma`), unless they are
  named. Every such fit printed xgboost’s
  `Parameters: { ... } are not used`.

- **`predict(iterationrange = )` on an `"xgboost"` model uses the rounds
  it documents, `c(start, end)` inclusive of both ends, on every xgboost
  version.** Before xgboost 3.0, which reads the end as exclusive,
  `c(1, 5)` predicted from four rounds; `ntreelimit` was translated the
  same way. A range past the model’s last round, or with its start after
  its end, is refused by name, where xgboost 3.x stopped with
  `Check failed: end <= model.BoostedRounds()`.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a multiclass
  `"nn"` model returns `NA` for a row with a missing predictor. The
  default type and `type = "class"` failed with
  `invalid subscript type 'list'`.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a `"deep"`
  model scores data without the response column, which failed with
  `object 'Species' not found`; returns `NA` for a row with a missing
  predictor, where three rows in came back as two; and reads factors on
  their training levels, so new data holding fewer levels is scored.
  keras no longer prints a progress bar on every prediction.

#### Metrics and evaluation

- **[`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  scores a model against the response its formula fits, the left-hand
  side evaluated on the scored rows.** It read the raw column, so a
  model of `log(mpg) ~ wt + hp` compared log-scale predictions with
  miles per gallon: rmse 18.1 on the training data for a fit whose
  residual rmse is 0.106, and with `set.seed(1)` `tl_cv(folds = 3)`
  reported rmse 18.0 and rsq -11.7 (now 0.133 and 0.731).
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md),
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  and the tuners score through
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md),
  so they move with it: with `set.seed(1)` and 3 folds,
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  on a tree of `log(mpg) ~ .` gave every `cp` the same 18.13 and now
  gives the log-scale 0.220.

- **[`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  reads the observed classes against the classes the model was trained
  on, and takes the model’s second class as the positive one.** A test
  split of `iris[iris$Species != "setosa", ]` still declares setosa, so
  every method stopped with “truth and estimate levels must be
  equivalent”, and
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md),
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md),
  the tuners, pipelines and
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  stopped the same way whenever a fold lacked a class: a logistic model
  of mtcars’ 0/1 `am` crashed `tl_cv(folds = 10)` for seeds 1 to 4. Rows
  of a class the model never saw, which a fold whose training rows
  missed a class produces, are left out with a warning naming the class.
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md),
  which has no model to consult, takes its classes from the levels of
  `predicted`, as [`predict()`](https://rdrr.io/r/stats/predict.html)
  returns them, plus any other class it finds; a prediction factor built
  by hand without a class the truth holds stopped with the same
  yardstick error, and those rows now count as errors. Predictions
  passed as character were read against every level the truth declared,
  so the values change: for a tree fitted on `iris[51:150, ]`, whose
  `Species` still declares setosa, precision and recall were a
  three-class macro average of 0.943 and 0.940 and are now the binary
  0.978 and 0.900.

- **`pr_auc` agrees with
  [`yardstick::pr_auc()`](https://yardstick.tidymodels.org/reference/pr_auc.html).**
  The area was integrated from the curve’s first finite point rather
  than from recall 0, so everything before that point was lost: a tree
  on two iris species reported 0.074 where yardstick gives 0.964, and a
  perfect ranking of 5 positives in 20 scored 0.8. With constant
  probabilities the `pr_auc` row was missing from the result. For more
  than two classes `pr_auc` was never computed and is now the average of
  the one-vs-rest areas, as `"auc"` is. The area in
  [`tl_plot_precision_recall()`](https://tidylearn.sheetsolved.com/reference/tl_plot_precision_recall.md)’s
  subtitle comes from the same corrected calculation.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  refuses to score when no row is left, where it returned `NaN` for
  `accuracy`, `rmse` and `mae` without a message. That covers `new_data`
  with no rows, such as an empty test split, and rows that are all
  dropped: for a missing response, a missing prediction (as wherever a
  predictor is missing) or a class the model was not trained on, and the
  message counts each. The error has class `tidylearn_no_scored_rows`.
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  catches it and leaves that fold out of the summary, with a warning
  giving the reason.
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  refuses an empty or wholly incomplete input the same way.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  and
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  refuse a metric name they do not compute, listing the ones they do.
  `tl_evaluate(model, metrics = "RMSE")`, a classification metric on a
  regression model and `tl_cv(metrics = "RMSE")` each returned a 0-row
  tibble without a message. `metrics = "sensitivity"` no longer adds a
  `recall` row it was not asked for.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  and
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  drop a row missing its observed class, prediction or probability
  before computing any metric, as the regression metrics already did. An
  `NA` in a predictor stopped `"auc"` with “‘predictions’ contains NA.”,
  and an `NA` response with “Not enough distinct predictions”, while
  `"accuracy"` dropped the same rows.

- `"auc"` and `"pr_auc"` are `NA`, with a warning, when the scored rows
  hold a single class. ROCR stopped on them with “Number of classes is
  not equal to 2”, which aborted
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  (with its default metrics, `set.seed(2)` and 5 folds on mtcars’ `am`)
  and, with `set.seed(1)`, `tl_tune_grid(metric = "auc", folds = 10)`.
  The fold now counts as unscored and the summaries average the others.
  For more than two classes, a class with no scored row gets an `NA`
  `auc_<class>` row and is left out of the averages, with a warning.

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  checks its arguments. `folds` must be a whole number between 2 and
  `nrow(data)`; `folds = 2.5` ran two folds without a message.
  `weights`, `subset`, `offset`, `foldid` and `strata` passed through
  `...` are refused by name, as
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  refuses them, since they hold one value per row and each fold fits a
  subset of the rows. A tree ignored them: on `mpg ~ wt + hp` with
  `set.seed(1)`, 3 folds and `weights = seq_len(32) / 32`, rmse was
  4.84, the same as without weights. A linear model failed with “..1
  used in an incorrect context”. An argument set to `NULL` is not
  refused. A one-column data frame, as `y ~ 1` uses, is kept a data
  frame in each fold, where it failed with “‘data’ must be a data
  frame”.

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  gives the warning that `method = "logistic"` converts a numeric 0/1
  response to a factor once per run. It came once per fold, so a
  logistic model of mtcars’ `am` at `folds = 5` warned five times.

- [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  refuses `predicted_probs` without a column for every class. A frame
  missing one was scored as a different problem: two classes with only
  the `a` column gave `auc` and `auc_a` of 1. A vector is refused with
  the shape it needs, where it failed with “argument is of length zero”.
  A matrix with a column per class is accepted, read by column name,
  where it failed with “subscript out of bounds”. The function warns
  when `thresholds` are ignored, for more than two classes or without
  probabilities, and when `"auc"` or `"pr_auc"` is asked for by name
  without probabilities. The input and output shapes of it and
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  are documented, including the `threshold` column `thresholds` adds and
  the placeholder row an unsupervised model returns.

#### Tuning and model selection

- A parameter set that fails on some folds can no longer be chosen as
  best. Its `mean_metric` was averaged over whichever folds survived, so
  it could win on the easier folds alone, and nothing in the results
  said so. The results now carry an `n_folds_ok` column, and only sets
  scored on every fold are eligible. If no set completes every fold, the
  best of the most complete sets is used, with a warning naming it. When
  every set fails in every fold the tuners stop with a message saying
  so, where they failed with `argument is of length zero`.

  The count and the warning cover folds whose fit succeeded but could
  not be scored, as well as failed fits, and the warning says which
  happened. A fold goes unscored when the metric is undefined there –
  precision where nothing is predicted positive, auc on a fold holding
  one class – or when none of its rows can be scored, such as one where
  every predictor is missing, which is warned about by fold and
  parameter set. In 0.5.0 such folds were dropped from the mean without
  a mention: with `metric = "precision"` on `mtcars`’s `am`,
  `folds = 10` and `set.seed(1)`, both sets were scored on 7 of the 10
  folds. A search in which every fit succeeded but no fold could be
  scored stops with a message saying so.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  decide the task from the response the formula computes, as
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  now does, and treat `method = "logistic"` as classification whatever
  type the response is stored as. `factor(am) ~ wt + hp` is tuned as a
  classification and defaults to accuracy. The tuners read the raw 0/1
  column, so in 0.5.0 that search was a regression scored by rmse (1.084
  for both sets on `mtcars`, `folds = 3`, `set.seed(1)`), and
  `metric = "accuracy"` was refused. They checked only for a factor or
  character response, so a 0/1 numeric response with
  `method = "logistic"` defaulted to `"rmse"`, which a logistic model
  never produces, and the search failed with
  `Metric "rmse" was not produced for this task`.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  refuse a `param_grid` or `param_space` element without a name, or two
  elements with the same name. An unnamed candidate vector reached
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  positionally, where it was discarded, so
  `param_grid = list(c(0.01, 0.1))` fitted the default tree twice (rmse
  4.656 for both on `mtcars`, `mpg ~ wt`, `folds = 3`, `set.seed(1)`)
  and returned an empty `best_params`.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  check `data`, `method`, `folds`, `metric`, `maximize` and `n_iter`
  before fitting anything, and the message names the argument.
  `metric = c("rmse", "mae")` failed with “the condition has length \>
  1”, `folds = 2.5` with an rsample error about `v`, and a misspelt
  `method` warned on every fit before failing with “Unknown method:
  trees”. `n_iter = 0` returned a fitted model with no error or warning,
  and `n_iter = 2.5` ran two iterations without a message; both are now
  errors.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md),
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md),
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  and
  [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  refuse `weights`, `subset`, `offset`, `foldid` and `strata` passed
  through `...`, as
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) and
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  do. Each holds one value per row and cannot follow the rows into a
  fold. In 0.5.0 a tree search ignored the weights, a linear search
  warned “variable lengths differ” on every fold and still returned a
  model, and a forest search failed with “invalid ‘length’ argument”.
  With `weights`,
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  failed with `..1 used in an incorrect context`, and
  [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  ran with keras ignoring them.
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  refuses the same arguments except `weights`, which it sets on the
  training `xgb.DMatrix()` and which `xgb.cv()` splits with the rows; a
  `subset` there reached xgboost as a parameter it does not use, and the
  tuning ran on every row.

- [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  uses a single value as given. `sample(20, 1)` draws from `1:20`, so
  `param_space = list(minsplit = 20, cp = c(0.01, 0.1))` tried
  `minsplit` values of 3, 10 and 6 with `seed = 1`. A logical was drawn
  from `c(TRUE, FALSE)` whatever was supplied, so `importance = TRUE`
  came out `FALSE` in three draws of four. Logicals are now drawn from
  the values given.
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  also accepts a list of candidates, such as
  `hidden_layers = list(10, c(20, 10))`, and `verbose = TRUE` no longer
  fails on a character parameter.

- Vector-valued grid candidates reach the model. The grid stores them in
  a list column, so the default `"deep"` grid passed
  `hidden_layers = list(c(10, 5))` and every multi-layer candidate
  failed.

- The large `"forest"` grid no longer includes `sampsize`, which
  randomForest reads as a number of rows and the grid gave as fractions
  from 0.5 to 1. For `method = "forest"` the tuners cap `mtry` at the
  number of predictors, counted the way randomForest counts them, with a
  warning; a matrix-valued term such as `poly(hp, 2)` counts once per
  column, since the forest is fitted on its columns. randomForest reset
  an oversized `mtry` in every fold while the results credited the value
  asked for. `y ~ . - id` does not count `id`: on
  `mtcars[, c("mpg", "wt", "hp", "qsec")]` plus an id column,
  `mpg ~ . - id` with `mtry = c(3, 4)` left 4 uncapped, randomForest
  reset it to 3 in every fold, and `best_params$mtry` reported 4 for a
  fit that used 3 (`folds = 3`, `set.seed(1)`). It is now capped at 3,
  and the duplicate set is evaluated once.

- `tl_default_param_grid("logistic")` returned the ridge `lambda` grid,
  which [`glm()`](https://rdrr.io/r/stats/glm.html) does not accept, so
  every fit in a search over it failed. It now returns an empty grid
  with a warning pointing to `"ridge"`, `"lasso"` and `"elastic_net"`.

- [`tl_default_param_grid()`](https://tidylearn.sheetsolved.com/reference/tl_default_param_grid.md)
  gives `"polynomial"` a `degree` grid, and says `"linear"` has nothing
  to tune; both used to warn “Unknown method”.
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  name a parameter given no candidate values, which was reported as
  every set failing or as `invalid first argument`. In
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  a whole-number range with equal ends, `c(20, 20)`, is the value 20 –
  it drew from 1 to 20 – and a non-whole one is refused with a message
  that no longer advises writing `c(20.5, 20.5)` as `c(20.5, 20.5)`. A
  forest `mtry` below 1 is raised to 1, as randomForest does, so the
  results report the value used.
  [`tl_plot_tuning_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tuning_results.md)
  on a one-parameter search explains that its default plot needs two
  parameters, where the message said `plot_type` must be one of
  “scatter”, … and had got “scatter”.

- [`tl_plot_tuning_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tuning_results.md)
  draws a parameter whose candidates are not single values, such as the
  deep grid’s `hidden_layers` or an rpart `parms` list, as a categorical
  one labelled as the verbose messages print it. Every plot type failed
  on them, the default scatter included. A categorical parameter with
  one value among the scored sets has importance 0, where the importance
  plot stopped with “contrasts can be applied only to factors with 2 or
  more levels”. The parallel plot draws one line per set: sets with tied
  scores shared a rank and were drawn as one line.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  give
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)’s
  note about the response once per search, from the final fit, and
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  does not repeat it, as
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  already did not. Every fold refit repeated it: a 2-set, 3-fold search
  on `mtcars` with `cyl ~ wt + hp` printed “Response ‘cyl’ has 3 unique
  numeric values” 7 times, and comparing two such models over 3 folds
  printed it 6 times. The warning that a numeric 0/1 response is
  converted to a factor for `method = "logistic"` is likewise given once
  per search, and not by
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)’s
  refits, which repeated it on every fold. Other warnings from the fits
  still come through.
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md),
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  and
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  give the note about the response once per call as well: on the first
  ten rows of `mtcars`, a linear model and a tree of `mpg ~ wt` over 3
  folds printed it 8 times,
  `tl_auto_ml(mtcars, cyl ~ wt + hp, time_budget = 10, cv_folds = 2)` 6
  times, and
  `tl_stratified_models(mtcars, mpg ~ wt + hp, k = 3, supervised_method = "linear")`
  twice (`set.seed(1)`).

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md),
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md),
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md),
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  and
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  warn once, before scoring, when every fold holds one row
  (`folds = nrow(data)`) and a metric being scored does not average to
  its leave-one-out value. rmse on one row is the absolute error, so the
  averaged rmse is the mean absolute error under rmse’s name: 2.52 for a
  linear `mpg ~ wt` on `mtcars` with `folds = 32`, the same as mae,
  where the leave-one-out rmse is 3.20. The warning says so, and that
  precision, recall, sensitivity, specificity and f1 are undefined on
  folds whose one row leaves nothing to divide by, and rsq, auc and
  pr_auc on every fold. The per-fold warnings from yardstick that it
  explains are no longer repeated. The scores are unchanged, and a run
  scoring only accuracy, mae, mse or mape gets no warning.

- **[`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  scores each two-class candidate by its own predictions.** Every
  candidate was scored as if it predicted the first class for every row,
  so on two-class `iris` with `set.seed(1)` and four folds every
  candidate scored 0.5, and the first in the grid always won. The
  candidates’ errors are 0.04 to 0.06.

- [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  refits its final model through
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md),
  so the model records the tuned settings in `$spec$args` and
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  refits each fold at them. It refitted at xgboost’s defaults: a model
  tuned to `max_depth = 1, eta = 0.01` scored fold for fold the same as
  the default model. Arguments in `...` that `xgb.cv()` takes go to it
  alone, and booster settings join every parameter set and the final
  fit; `showsd = FALSE` had reached the final `xgb.train()`, which
  warned that it did not recognise it. Case `weights` apply to the
  cross-validation and the final fit.

- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  refits each model with the arguments it was built with. It refitted
  from the formula and method alone, so every model was scored at its
  method’s defaults: a tree with `cp = 0.5` and one with
  `cp = 0.0001, minsplit = 2` produced identical fold scores. Models now
  record their fitting arguments in `$spec$args`; arguments passed to
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  override them. A model fitted with `weights`, `subset`, `offset`,
  `foldid` or `strata` is refused, since those hold one value per
  training row and cannot follow the rows into a fold. Those are
  recorded by name only, in `$spec$per_row_args`, so a model does not
  carry a second copy of its weights.

- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  refuses a model that a refit from its formula, method and arguments
  would not reproduce: one built by
  [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  or
  [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md),
  or a
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  candidate fitted on PCA scores or cluster labels. A semi-supervised
  model trained on 15 labels was scored as a tree fitted on every
  training-fold label (accuracy 0.947 on `iris`, `folds = 3`,
  `set.seed(1)`), and an anomaly-aware model with `action = "flag"`
  failed with “variable lengths differ (found for ‘is_anomaly’)”.

- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  checks its arguments before fitting anything. A metric name it does
  not compute for the task is refused with the message
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  gives: a misspelt `"acuracy"` was dropped from the summary without a
  message, and `metrics = "rmse"` on a classification model returned an
  empty summary. `tl_compare_cv(mtcars, list())` failed with “argument
  is not interpretable as logical”, a single model passed without
  [`list()`](https://rdrr.io/r/base/list.html) was read as three models
  that were not tidylearn models, and an unsupervised model failed the
  task check; each now gets a message saying what to pass.

- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  refuses repeated model names and names an unnamed entry `Model_<i>`,
  numbered on (`Model_1.1`) if the caller already chose that name.
  Results are keyed on the name, so `list(a = m1, a = m2)` pooled both
  models’ folds into one summary row, and a partly named list produced a
  model called `""`.

- [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  reports `NA` where it has no value. A fold none of whose rows can be
  scored, such as one where every predictor is missing, has `NA` for
  each metric in `fold_metrics`, with a warning naming the fold and the
  model; in 0.5.0 it scored `NaN` and was left out of the summary
  without a message. A metric with no value on any fold summarises as
  `NA`: its `mean_value` was `NaN`, and its `min_value` and `max_value`
  were `Inf` and `-Inf` with a warning each.
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  reports such a metric’s mean as `NA` too, as its note said it did: rsq
  on `data.frame(x = 1:10, y = 3)` with `folds = 5` summarised as mean
  `NaN`.

- [`tl_step_selection()`](https://tidylearn.sheetsolved.com/reference/tl_step_selection.md)
  with `direction = "forward"` or `"both"` works with a formula using
  `.`, and keeps a transformed response, the formula’s
  [`offset()`](https://rdrr.io/r/stats/offset.html) terms and its
  intercept setting. It returned the intercept-only model for any
  formula using `.`: [`step()`](https://rdrr.io/r/stats/step.html)
  expanded the dot against the starting model’s `1`, which left no terms
  to add. The formula is now expanded against the data first. The
  starting model was built from the response’s variable name, so
  `log(mpg) ~ wt + hp + qsec` selected a model of `mpg`,
  `mpg ~ wt + hp + offset(qsec)` returned `mpg ~ wt + hp` with no
  offset, and `mpg ~ wt + hp - 1` came back with an intercept. The last
  two now give `mpg ~ wt + offset(qsec)` and `mpg ~ wt - 1`, as backward
  selection did.

- **[`tl_step_selection()`](https://tidylearn.sheetsolved.com/reference/tl_step_selection.md)
  fits every candidate model on the rows with no missing value in the
  formula’s variables, and a message gives the count.** A variable with
  missing values changed the row count as it entered or left, so
  backward and both-direction selection stopped with “number of rows in
  use has changed: remove missing values?”, and forward selection
  compared the candidates on the complete rows, then returned a model
  fitted on more: with three missing values in a noise column added to
  `mtcars`, it compared on 29 rows and fitted on 32. The model keeps the
  data passed in, and the rows left out are recorded in the fit’s
  `na.action`.

- `tl_step_selection(criterion = "BIC")` penalises by the rows the model
  used. `log(nrow(data))` counted rows
  [`lm()`](https://rdrr.io/r/stats/lm.html) dropped for missing values,
  which could change the model selected. Forward and `"both"` selection
  also find a variable the formula takes from the caller’s environment.

- [`tl_step_selection()`](https://tidylearn.sheetsolved.com/reference/tl_step_selection.md)
  refuses a categorical response, saying that it selects linear models.
  A factor such as `Species`, or one the formula makes with
  `factor(am)`, had its codes fitted until
  [`step()`](https://rdrr.io/r/stats/step.html) stopped with “AIC is
  -infinity for this model, so ‘step’ cannot proceed”.

- [`tl_test_model_difference()`](https://tidylearn.sheetsolved.com/reference/tl_test_model_difference.md)
  accepts `test = "wilcox.test"` as `"wilcox"`. The diagnostics vignette
  recommended `"wilcox.test"` at small fold counts, a value the function
  rejected; it now uses `test = "wilcox"`. The vignette and the help
  page give the fold count the exact signed-rank test needs: the
  smallest two-sided p-value over n folds is 2 / 2^n, which is 0.0625 at
  the vignette’s 5 folds, so it takes 6 to reach p \< 0.05.

- **[`tl_test_model_difference()`](https://tidylearn.sheetsolved.com/reference/tl_test_model_difference.md)
  pairs the two models’ scores by fold and uses the folds both have a
  value for, in the test and in `mean_diff`.** `mean_diff` was each
  model’s mean over its own scored folds: for fold scores (0.5, 0.6, NA,
  0.8) and (0.4, 0.7, 0.9, 0.6) it was 0.017, where the mean paired
  difference is -0.067. A comparison with fewer than two such folds has
  an `NA` p-value, with a warning;
  [`t.test()`](https://rdrr.io/r/stats/t.test.html) stopped there, which
  discarded every other metric’s result.

#### Pipelines, AutoML and integration

- **[`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  standardises a numeric predictor only where the model the formula
  describes stays the same.** It leaves alone a column used inside a
  function call, such as [`log()`](https://rdrr.io/r/base/Log.html),
  [`poly()`](https://rdrr.io/r/stats/poly.html),
  [`I()`](https://rdrr.io/r/base/AsIs.html) or
  [`offset()`](https://rdrr.io/r/stats/offset.html); every column of a
  formula without an intercept; and the columns of an interaction whose
  lower-order terms are not all in the formula. Other plain terms are
  standardised as before. The formula was evaluated on the standardised
  columns, so `mpg ~ log(hp) + wt` on `mtcars` took the log of negative
  z-scores, warned “NaNs produced”, fitted the final
  [`lm()`](https://rdrr.io/r/stats/lm.html) on 15 of the 32 rows (a
  `log(hp)` coefficient of -0.22, where
  [`lm()`](https://rdrr.io/r/stats/lm.html) gives -5.92), and
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  returned `NaN` for 5 of the first 6 cars.
  `mpg ~ wt + offset(0.05 * hp)` applied the offset per standard
  deviation of `hp`, and its predictions were up to 8.5 mpg from those
  of [`lm()`](https://rdrr.io/r/stats/lm.html) on the same formula;
  `mpg ~ wt - 1` fitted a line through a different origin. On data
  without missing values, a `"linear"` or `"logistic"` pipeline now
  predicts what [`lm()`](https://rdrr.io/r/stats/lm.html) or
  [`glm()`](https://rdrr.io/r/stats/glm.html) predicts for the same
  formula.

- **[`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  with `validation = "split"` stores the training split’s statistics in
  `$preprocessing_stats` and the preprocessed training split in
  `$processed_data`, since the models it keeps were fitted there.** Both
  held full-data values, so
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  centred `wt` on 3.217 (all 32 rows of `mtcars`) for a model fitted
  under 3.279 (its 22 training rows), and its predictions for the test
  rows were up to 0.685 mpg from the ones the model was scored on
  (`set.seed(3)`). They now match.

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  and
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  take the task from the response the formula computes, as
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  does. They read it off the raw column, so `factor(am) ~ wt + hp` on
  `mtcars` ran as a regression of `am`’s 0/1 codes: the pipeline scored
  its tree by rmse, and AutoML chose the task “regression” and ranked
  its candidates by rmse. Both now fit and score a classification.

- **Best-model selection in
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  treats `sensitivity`, `specificity` and `pr_auc` as
  higher-is-better.** They were ranked lowest-first, so
  `best_metric = "sensitivity"` on two-class iris chose a `cp = 1` stump
  scoring 0.400 over a tree scoring 0.898 (`set.seed(10)`). The
  `higher_better` column of
  [`tl_compare_pipeline_models()`](https://tidylearn.sheetsolved.com/reference/tl_compare_pipeline_models.md)’s
  plot data does the same, and no longer marks a multiclass run’s
  `auc_<class>` rows lower-is-better. The pipeline, its comparison plot
  and the
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  leaderboard now take the direction from one list.

- [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  warns about a cross-validation fold with no row it can score, such as
  one where every response is missing, and averages the model’s scores
  over the other folds. The fold’s scores came back `NaN` and were
  dropped from the average without a message. With
  `validation = "split"` an unscorable test set is now an error, since
  there is nothing else to score; its scores were `NaN`, and the run
  warned that every value was `NA` and returned the first model.

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  refuses an argument it does not take, and
  `preprocessing = list(dummy_encode = FALSE)`. A misspelt argument was
  swallowed by `...`: `evalution = list(cv_folds = 2)` left `cv_folds`
  at 5. `dummy_encode = FALSE` ran the same models as `TRUE`, because
  each model’s fitting function encodes factors itself.

- [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  and
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  give the logistic “Converting response variable to factor” warning
  once per call. They refit once per fold, model and cluster, and
  repeated it each time: two logistic specs over three folds on a 0/1
  response warned eight times.

- [`predict()`](https://rdrr.io/r/stats/predict.html) and
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  on one of
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)’s
  `pca_*` candidates accept the model’s own stored data, which holds the
  component scores in place of the raw columns, and
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  and [`summary()`](https://rdrr.io/r/base/summary.html) score that data
  for any model fitted on engineered features. Passing it back
  (`tl_evaluate(model, model$data)`, or a plot that defaults to it)
  projected it a second time and failed with “PCA was fitted on 4
  column(s) … but new_data is missing”. When
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  fell back to training scores, those candidates dropped off the
  leaderboard.

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  checks `task` and `metric` before fitting anything. `metric` must be
  one of the task’s metrics that
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  computes: `"sensitivity"` and `"specificity"` were refused only after
  every candidate had been fitted, and `"roc_auc"`, `"kap"`, `"logloss"`
  and `"mlogloss"` were accepted but never computed, so every score was
  `NA` and the first model came back. `task` must be `"auto"`,
  `"classification"` or `"regression"` and agree with the response:
  `task = "Classification"` fitted the regression methods to a factor
  and scored every one `NA`, and
  `tl_auto_ml(mtcars, am ~ wt + hp, task = "classification")`, which
  ran, is refused because `am` is numeric and every candidate but
  logistic regression fits it as a regression; write the response as
  `factor(am)` instead. A formula response that is not a column of
  `data` is refused as well; every fit used to fail, leaving an empty
  leaderboard.

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  builds its PCA and cluster candidates from the predictors the formula
  names, and fits the cluster candidates with the cluster column added
  to the formula’s terms. The rotation and the k-means centres were
  fitted on every column but the response, so `Species ~ . - id` on iris
  put the excluded row id into both, and iris is sorted by class, so the
  id carries the class. With an explicit right-hand side,
  `Species ~ Sepal.Length + Sepal.Width`, the PCA used all four
  measurements, and the clustered candidates left the cluster column out
  of their formula, so `clustered_tree` predicted exactly what
  `baseline_tree` did. The number of clusters for a classification task
  counts the observed classes; a missing response value was counted as
  one.

- **[`tl_explore()`](https://tidylearn.sheetsolved.com/reference/tl_explore.md)
  keeps `max_components` principal components.** It kept every
  component, so `max_components = 2` on iris gave 4, and the default on
  `mtcars` gave 11 where it now gives 5. `k_range` must hold whole
  numbers from 2 to one less than the number of rows; `k_range = 1:3`
  failed with “incorrect number of dimensions”.

- [`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)
  takes a string formula and an explicit right-hand side, fits its PCA
  on the predictors the formula names, and accepts
  `pretrain_method = "pca"` only. A string gave “Response variable ‘NA’
  not found”; `Species ~ Sepal.Length + Petal.Length` failed with
  “object ‘Sepal.Length’ not found”, because the tree was fitted on the
  component scores under the raw-column formula; the documented
  `"autoencoder"` gave “Unknown method”; and an `"mds"` fit could not
  project the rows [`predict()`](https://rdrr.io/r/stats/predict.html)
  was given.

- [`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)
  names an unusable `method` or `n_components`: `method = "kmeans"`
  failed with “object ‘transformed’ not found”, and `n_components = 6`
  on iris with “Elements PC5 and PC6 don’t exist”.
  [`tl_compare_pipeline_models()`](https://tidylearn.sheetsolved.com/reference/tl_compare_pipeline_models.md)
  names a requested metric the run did not score, which failed inside
  ggplot2’s faceting. [`summary()`](https://rdrr.io/r/base/summary.html)
  of a run pipeline printed the best model’s summary twice; it prints it
  once.

- [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  refuses a numeric response. Labels are propagated by majority vote
  within a cluster, and the response was passed through
  [`factor()`](https://rdrr.io/r/base/factor.html), so `mpg ~ .` was
  fitted as a classification with one class per distinct value. The
  response is the one the formula computes, so
  `factor(am) ~ wt + hp + qsec` is accepted and `log(mpg) ~ wt + hp`
  refused as numeric. A computed logical such as `I(mpg > 20)`, which
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  fits as a regression, is refused with a pointer to
  [`factor()`](https://rdrr.io/r/base/factor.html), and so is a response
  that does not compute the propagated labels back from the column’s
  values, such as `cut(mpg, 2)`, whose breaks follow the column’s range.

- [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  warns about rows whose cluster holds no labelled observation, and
  leaves them out of training. They were given `NA` labels and dropped
  when the model was fitted, without a message; with six labels from two
  iris classes, 53 of 150 rows went that way. The count is in
  `$semisupervised_info$n_unlabelled_dropped`. Labelled rows whose label
  is missing are left out of the vote. A cluster whose labelled rows all
  had missing labels gave all its rows the response’s first level: with
  the labelled setosa rows set to `NA` and levels
  `c("virginica", "versicolor", "setosa")`, the 45 unlabelled setosa
  rows were trained as virginica (`set.seed(2)`), and for a character
  response the same call failed with “Can’t combine NULL and non NULL
  results”. Those rows are now left out of training too, and the warning
  gives unlabelled rows and rows with a missing label separately. k is
  set to the number of classes the labelled rows carry, and labelled
  rows from fewer than two classes are refused. A logical
  `labeled_indices` and a string formula are also accepted now; the
  logical was matched as the positions 0 and 1.

- **[`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  keeps the response’s level order.** The pseudo-labels were re-sorted
  alphabetically, so with levels `c("virginica", "versicolor")` the
  model came back with versicolor first and virginica as the positive
  class: a logistic model’s `.pred` for the first versicolor row was
  0.001, where it is now 0.999 (`set.seed(1)`).

- **[`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md),
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  and
  [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)
  cluster, or look for anomalies, on the predictors the formula names,
  so their results change for any formula that leaves columns of `data`
  out.** They used every column but the response, so a column the
  formula excludes took part: with a row id added to iris and
  `Species ~ . - id`, the id was one of the k-means columns.

- `tl_add_cluster_features(method = "hclust")` takes `k`. It was passed
  on to the tree fit, which has no `k`, so every value but the fallback
  failed with `unused argument (k = 4)`.
  `tl_semisupervised(cluster_method = "hclust")` failed the same way on
  every call, and now cuts the tree.

- [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  takes `cluster_method = "kmeans"`, `"pam"`, `"clara"` or `"hclust"`,
  whose tree is cut at `k`. `"hclust"` and `"dbscan"` failed with
  “unused argument (k = 2)”, and a `"pam"` fit could not predict even
  its own training rows. The training rows’ assignments are stored in
  `$clusters`, which [`predict()`](https://rdrr.io/r/stats/predict.html)
  uses when `new_data` is `NULL`. `"dbscan"`, which picks its own number
  of clusters, is refused by name here and in
  [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md),
  where it failed the same way.

- [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  handles a cluster whose rows all hold one class. Every classifier
  refuses a one-class response, so
  `tl_stratified_models(iris, Species ~ ., k = 3)` failed
  (`set.seed(1)`): k-means puts the 50 setosa rows in a cluster of their
  own. That cluster now gets no model and is listed in
  `$single_class_clusters`,
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns its class
  for its rows (probability 1 under `type = "prob"`), and the other
  clusters’ models are built as before. This holds for a response the
  formula computes, such as `factor(am)`, whose single-class clusters
  were not found.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  result returns what each cluster’s model returns for the requested
  `type`. With `type = "prob"` it returned only `.cluster`, with a
  warning per row. Class levels and probability columns follow the
  classes in the training data: they had followed whichever rows came
  first, so the second level – the positive class – changed with row
  order, and a class one cluster never saw had probability `NA` where it
  now has 0.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  result works on data without the response column. It selected the
  response out by name and failed with `Element mpg doesn't exist`.

- **`tl_anomaly_aware(action = "downweight")` now changes the model.**
  With the default `"tree"` the weights went to `rpart.control()`, which
  ignored them, so the fit was identical to the unweighted tree; with
  `"linear"`, `"boost"` and `"nn"` the call failed with
  `..1 used in an incorrect context`; `"xgboost"` warned “Passed
  unrecognized parameters: weights” and fitted without them, so its
  downweighted fits change; and `"svm"` ignored the weights. It now
  requires a `supervised_method` that applies case weights – `"linear"`,
  `"polynomial"`, `"logistic"`, `"tree"`, `"ridge"`, `"lasso"`,
  `"elastic_net"`, `"forest"`, `"boost"`, `"nn"` or `"xgboost"` – and
  refuses `"svm"` and `"deep"`, which take no case weights. The
  `"boost"`, `"nn"` and `"xgboost"` fits match
  [`gbm::gbm()`](https://gbm-developers.github.io/gbm/reference/gbm.html),
  [`nnet::nnet()`](https://rdrr.io/pkg/nnet/man/nnet.html) and
  [`xgboost::xgb.train()`](https://rdrr.io/pkg/xgboost/man/xgb.train.html)
  given the same weights. A forest reads them as sampling weights.

- `tl_anomaly_aware(action = "flag")` adds the flag to the formula as
  written. It rebuilt the formula from its variable names, so
  `Species ~ . - Sepal.Width` put `Sepal.Width` back in as a predictor
  and `mpg ~ poly(wt, 2)` was fitted as `mpg ~ wt`.

- [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)
  stops when DBSCAN marks every row as noise, naming `eps`, `minPts` and
  the unscaled predictors. At the default `eps = 0.5`, all 32 rows of
  `mtcars` are noise for `mpg ~ wt + hp`: `action = "remove"` failed in
  [`lm()`](https://rdrr.io/r/stats/lm.html) with “0 (non-NA) cases”,
  `"downweight"` failed with “..1 used in an incorrect context”, and
  `"flag"` returned a model whose `is_anomaly` coefficient was `NA`.

- [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md)
  names an invalid `action`, which failed with
  `object 'model' not found`, and no longer documents an
  `"isolation_forest"` method it never had.

#### Preprocessing, interactions and diagnostics

- `tl_split(stratify = )` splits rows whose stratify value is missing.
  [`split()`](https://rdrr.io/r/base/split.html) drops an `NA` group, so
  those rows were never drawn and all went to test. They now form a
  stratum of their own.
  [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  on a one-column data frame also returns data frames in place of bare
  vectors.

- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  refuses a `stratify` that is not the name of one column.
  `stratify = c("am", "vs")` failed with `the condition has length > 1`,
  and an unknown column is now named in the error.

- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  refuses a `prop` outside (0, 1). `prop = 1.5` split 32 rows 31/1,
  because each side is kept to at least one row.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  imputes with the method it names. `"mode"` and `"knn"` both used the
  mean while the message reported the method asked for; `"mode"` is now
  implemented and anything else is refused. A missing value in a factor
  or character column was never imputed, and with more than two levels
  it broke one-hot encoding with
  `Can't recycle ..1 (size 150) to match ..2 (size 148)`. Categorical
  gaps now take the column’s most frequent value.

- `tl_prepare_data(remove_correlated = TRUE)` removes the feature that
  clears a correlated group, where it removed the ones around it. Every
  pair was decided separately against a half-zeroed matrix, so for a
  chain `x1 - x2 - x3` it dropped `x1` and `x3` and kept `x2`, the one
  feature correlated with both. Features are now removed one at a time,
  the most correlated pair first.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  processes only the formula’s predictors. It read the formula for the
  response alone, so `y ~ . - id` one-hot encoded `id` into twenty
  columns. A column the formula excludes is now returned unchanged.
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  shares that formula reading to record the training levels of factor
  predictors, and for a formula using `.` it recorded none.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  refuses two inputs it used to misread. A `scale_method` it does not
  have, such as `"zscore"`, reported “Scaling numeric features using
  method: zscore” and returned the data unscaled. A formula with no
  response, such as `~ x1 + x2`, took its first predictor as the
  response and passed it through unscaled.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  refuses a `correlation_cutoff` that is not a single number greater
  than 0 and at most 1. No correlation exceeds 1, so a cutoff of 95,
  meant as a percentage, made `remove_correlated = TRUE` remove nothing
  without a message.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  no longer stops on an entirely missing numeric column with
  `remove_zero_variance = FALSE`, on an entirely missing factor
  (`Can't recycle ..1 (size 6) to match ..2 (size 0)`), on a single row
  or a column holding an `Inf` (both
  `Selections can't have missing values`), or on a grouped tibble
  (`mpg must be size 11 or 1, not 32`). The missing factor is left as it
  is, a column whose spread is not finite is left unscaled, and a
  grouped tibble is prepared as a whole and returned ungrouped. A
  response that is not a column of the data is named in the error:
  `mgp ~ .` failed with dplyr’s `Element mgp doesn't exist`. Its
  documentation says the statistics come from the data passed in, so
  preparing before splitting lets test rows shape them.

- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md)
  reports encoding, and records an encoding step, only for the variables
  it splits into dummy columns. A two-level factor is left as it is, yet
  one alone gave “Encoding 1 categorical variables”. Imputing and
  scaling are likewise reported and recorded only when a column is
  filled or scaled.

- [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
  honours `exclude_vars`. It computed the reduced predictor set and
  never used it, so with a strong `a:z` effect `exclude_vars = "z"`
  still returned `y ~ a + b + z + a:z`. A name in `exclude_vars` that is
  not a predictor in the formula is now an error. With a `y ~ .` formula
  the function failed in its testing step, and now works. When every
  pair is already in the formula it returns the model as specified, with
  a message.

- `tl_auto_interactions(top_n = 0)` adds no interaction.
  `significant[1:0, ]` selected the first row, so `top_n = 0` still
  returned `y ~ a + b + z + a:z`. `top_n` must now be a single whole
  number of 0 or more. The `interaction_tests` and
  `selected_interactions` attributes are set on every returned model, as
  documented; they were missing when nothing was significant or nothing
  was left to test.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  and
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  accept a model fitted with `y ~ .`, which failed with
  `Variables not found in model formula`.
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  also stopped repeating grid blocks when quartiles of `by_var` tie:
  `mpg ~ wt * cyl` gave five slopes for three values of `cyl`, and now
  gives one each, labelled with the quartiles it stands for.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  and
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  report a classification model on the probability of its second class.
  A forest or tree classifier predicts the class label by default, so
  `fit` was a factor and the slopes were fitted to class labels (0.390
  to 0.482 for a two-class iris forest fitted after `set.seed(1)`, under
  a run of warnings), and a contour plot of two numeric variables failed
  with `'range' not meaningful for factors`. A model with more than two
  classes is refused. A logistic model’s values are unchanged.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  and
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  refuse a model variable that shares a name with a column they write
  their results to: `fit`, `by_value` and `by_label`, and with intervals
  `lower` and `upper`, for the effects; `prediction`, and with a band
  `.lower` and `.upper`, for the plot. The results overwrote the
  variable: `var = "fit"` on `mpg ~ fit * hp` failed with
  `subscript out of bounds`, a `by_var` named `fit` or a held variable
  named `lower` came back replaced by the predictions or the interval,
  and a variable named `prediction` was plotted as the predictions
  themselves.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  refuses a constant `var`, which failed with `subscript out of bounds`,
  and
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  refuses a `type` argument: `type = "class"` drew class codes over a
  probability band.

- `tl_plot_interaction(confidence = TRUE)` draws the confidence band.
  The band came from
  [`predict()`](https://rdrr.io/r/stats/predict.html), which returns no
  interval, so none was ever drawn. It now comes from the `lm` or `glm`
  fit, for numeric by categorical plots; a model without standard errors
  gets a message.

- [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md)
  accepts a `.` formula and a string formula, and stops with a message
  naming the formula’s predictors when a type filter leaves no pairs.
  These failed with messages such as `undefined columns selected` and
  `argument 1 is not a vector`. It no longer re-tests a pair whose
  interaction the formula already has, which came back as a row of `NA`,
  and it refuses a formula with no response, whose first predictor it
  treated as the response.

- [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md)
  and
  [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
  no longer treat the variable inside an
  [`offset()`](https://rdrr.io/r/stats/offset.html) as a candidate
  predictor. For `mpg ~ wt + hp + offset(log(disp))`,
  [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md)
  also tested `hp:disp` and `wt:disp`, and
  [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
  could add an interaction with the offset’s variable, returning
  formulas such as `y ~ a + b + a:e + offset(log(e))`.

- [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md)
  and
  [`tl_auto_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_auto_interactions.md)
  refuse a categorical response.
  [`lm()`](https://rdrr.io/r/stats/lm.html) cannot fit one, and a
  two-class factor response returned a row of `NaN` after a run of
  warnings. A logical response is still fitted as 0/1.

- A non-syntactic column name, such as `car weight`, works in
  `tl_detect_outliers(method = "cook")`, as `var` in
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md),
  and as `var1` or `var2` in
  [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md).
  Each pasted the name into formula text, which failed with
  `unexpected symbol`.

- **[`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  reads a factor’s generalised VIF on the scale of an ordinary one.**
  [`car::vif()`](https://rdrr.io/pkg/car/man/vif.html) returns a table
  of GVIF, Df and GVIF^(1/(2\*Df)) when a term has more than one degree
  of freedom, and the maximum was taken over the whole table:
  `mpg ~ wt + carb` with `carb` a six-level factor reported “Maximum
  VIF: 5” and multicollinearity, from `carb`’s 5 df, while both GVIFs
  are 1.6. The check now uses GVIF^(1/Df), which equals the VIF for a
  one-df term.

- **[`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  no longer tests a logistic model for normal residuals or constant
  variance, neither of which logistic regression assumes.** Shapiro-Wilk
  on the deviance residuals of `am ~ wt + hp` gave p = 0 with advice to
  transform the response, and
  [`lmtest::bptest()`](https://rdrr.io/pkg/lmtest/man/bptest.html)
  tested a linear probability model. Both checks now have a `NULL` check
  and a note saying why, so that model goes from 3 violations in 6
  checks to 2 in 4.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  counts a check its test could not decide as neither satisfied nor
  violated. A model with fewer than four distinct fitted values leaves
  the linearity check `NA`, and the verbose summary failed with
  `missing value where TRUE/FALSE needed`; with `verbose = FALSE` the
  status read “NA assumption(s) appear to be violated”. Such a check is
  now reported as UNKNOWN. A one-factor model such as `len ~ supp` has
  two fitted values that floating-point noise made look like four, so
  its linearity test ran and reported p = 1; it is now `NA` too.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  always returns its multicollinearity check. When
  [`car::vif()`](https://rdrr.io/pkg/car/man/vif.html) could not run,
  the fallback result was assigned inside the error handler and lost, so
  `mpg ~ wt` came back with no `multicollinearity` element and five
  checks instead of six. The fallback counts the fit’s terms, so a
  `y ~ .` formula is no longer read as one predictor.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  no longer advances the session’s random number stream. It called
  [`car::durbinWatsonTest()`](https://rdrr.io/pkg/car/man/durbinWatsonTest.html),
  which bootstraps a p-value the function never read. The Durbin-Watson
  statistic is now computed directly, matches
  [`lmtest::dwtest()`](https://rdrr.io/pkg/lmtest/man/dwtest.html), and
  is reported whether or not car is installed.

- [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md)
  works on a rank-deficient fit and on one fitted with
  `na.action = na.exclude`.
  [`dfbetas()`](https://rdrr.io/r/stats/influence.measures.html) has no
  column for an aliased coefficient, and looping over
  [`coef()`](https://rdrr.io/r/stats/coef.html) failed with
  `subscript out of bounds`; under `na.exclude` every measure is padded
  back to all rows, and building the result failed with
  `arguments imply differing number of rows: 30, 32` (`116, 153` for
  `Ozone ~ Temp + Wind` on `airquality`).
  [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  failed on both fits too: with `subscript out of bounds` on the
  rank-deficient one and `residuals include missing values` under
  `na.exclude`. The default thresholds count the coefficients the fit
  estimated.

- **[`tl_detect_outliers()`](https://tidylearn.sheetsolved.com/reference/tl_detect_outliers.md)
  flags and counts the right rows when a value is missing.** With
  `method = "cook"` the fit dropped the incomplete row and its 31
  distances were recycled into a 32-row flag matrix: one missing `wt` in
  `mtcars` flagged nine rows where four have a Cook’s distance above
  4/32. Those four are now flagged, and the incomplete row’s flags are
  `NA`. For every method `any_outlier` and `outlier_counts$total` skip a
  missing flag: on `airquality` the total was `NA` while
  `outlier_indices` listed five rows, and it is now 5. A single row no
  longer fails with `dim(X) must have a positive length` under `"iqr"`
  or `"z-score"`.

- [`tl_diagnostic_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_diagnostic_dashboard.md)
  refuses a model whose fit is not an `lm` or `glm` before drawing
  anything. A tree failed partway through with
  `no applicable method for 'rstandard' applied to an object of class "rpart"`.

#### Unsupervised learning

- A one-sided formula for an unsupervised method names columns only. A
  term such as `log(Sepal.Length)` is an error naming it; PCA used to
  run on the raw column instead, centred at 5.84, the mean of
  `Sepal.Length` itself. `~ . - x` now means every numeric column but
  `x`, where it failed with “undefined columns selected”. A name that is
  not a column of the data, as in `~ wt + zz`, or `~ . + z` with `z` in
  the caller’s session, is an error that names it; it failed with
  “undefined columns selected” too.

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  warns, naming the column, when an unsupervised formula names a
  non-numeric column the method cannot use: any such column for
  `"kmeans"`, `"pca"` and `"clara"`, and for `"mds"`, `"pam"`,
  `"hclust"` and `"dbscan"` unless the distance is Gower.
  `~ Sepal.Length + Species` clustered on Sepal.Length alone, and PCA
  returned two components for three named columns, without a message.
  The fit is unchanged. For the four methods that take a distance the
  warning points to the Gower distance, which uses the column; a `~ .`
  formula still means the numeric columns and stays quiet.

  A non-numeric column selected by `cols` is left out with a warning
  naming it wherever the method uses numeric columns only, which is
  everywhere but a Gower distance, and a selection with no numeric
  column is refused by name.
  [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md),
  [`tidy_pca()`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md),
  [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md)
  and
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  failed on such a column inside their backends
  (`NA/NaN/Inf in foreign function call`,
  `'x' must be numeric or complex`, `x has to be a numeric matrix`,
  `all data in x has to be numeric`),
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md),
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  and
  [`tidy_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_dist.md)
  dropped it from the distance without a message while
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)’s
  `$medoids` and
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)’s
  `$data` kept it (both now leave it out), and
  `tidy_dist(iris, cols = "Species")` returned a distance matrix of
  `NA`, which
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)
  and
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  then refused as missing values.
  [`tidy_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_dist.md),
  and so
  [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md),
  refuse data with no numeric column in the same words.

- **The unsupervised functions ignore dplyr grouping.** Selecting the
  numeric columns of a grouped tibble brought the grouping variable back
  (“Adding missing grouping variables”), so on `group_by(iris, Species)`
  [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  clustered on Species’ factor codes and partitioned 14 of 150 rows
  differently,
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)
  and
  [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)
  worked from distances 1.118 times too large (Species coerced to `NA`),
  `tl_model(method = "kmeans")`,
  [`calc_wss()`](https://tidylearn.sheetsolved.com/reference/calc_wss.md),
  [`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)
  and
  [`tidy_silhouette_analysis()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette_analysis.md)
  failed with `NA/NaN/Inf in foreign function call`,
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  with `all data in x has to be numeric`,
  [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md)
  with `x has to be a numeric matrix`, and
  [`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)
  with `'x' must be numeric or complex`. A grouped tibble now gives the
  fit of the same data ungrouped, through
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  and the `tidy_*()`, distance, k-NN and validation helpers.

- **[`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  and `tl_model(method = "dbscan")` cluster on the `distance` asked
  for.** It was ignored: on `iris[, 1:4]` with `eps = 0.5` and
  `minPts = 5`, `distance = "manhattan"` gave the Euclidean answer of 2
  clusters and 17 noise points, where
  [`dbscan::dbscan()`](http://michael.hahsler.net/dbscan/reference/dbscan.md)
  on Manhattan distances finds 3 clusters and 91 noise points. Any
  method [`stats::dist()`](https://rdrr.io/r/stats/dist.html) accepts
  works, and `"gower"`.

- **Gower distance uses the factor columns the caller selected.**
  `tidy_hclust(distance = "gower")` and
  `tl_model(method = "hclust", distance = "gower")` kept only the
  numeric columns, with or without a formula, so even a factor named in
  `~ num + fac` never reached the distance. For PAM with
  `metric = "gower"`, `~ .` expanded to the numeric columns, and the fit
  disagreed with the no-formula one on 8 of 20 rows of a mixed example.
  With Gower, no selection and `~ .` mean every column, `~ . - x` every
  column but `x`, and the heights and clusterings match
  [`cluster::daisy()`](https://rdrr.io/pkg/cluster/man/daisy.html)’s
  distances. `tidy_dbscan(distance = "gower")` takes every column the
  same way.

- **[`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  gives `NA` for a pair of rows with no variable observed in both, as
  [`cluster::daisy()`](https://rdrr.io/pkg/cluster/man/daisy.html)
  does.** It gave 0, so
  `tidy_gower(data.frame(x = c(1, NA, 3), y = c(NA, 2, 5)))` called rows
  1 and 2 identical and
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  grouped them together.
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md)
  and
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)
  now stop on undefined distances and name the pairs of rows, where they
  clustered them as identical.

- [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  names what is missing when its data has missing values: on
  coordinates, the columns holding them; on any other distance, the
  pairs of rows with no variable observed in both. `dbscan()` stopped
  with `data/distances cannot contain NAs for dbscan (with kd-tree)!`.
  Missing values that leave every pair a shared variable cluster on a
  distance that tolerates them, such as `"manhattan"`.

- **[`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)
  leaves DBSCAN noise (cluster 0) out of every measure, and reports it
  in a new `n_noise` column, so
  [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  gains the column too.** Only `k` excluded noise before. For
  `tidy_dbscan(iris[, 1:4], eps = 0.5, minPts = 5)`, whose clusters hold
  49 and 84 rows, it reported `min_size` 17 (the noise count),
  `avg_size` 50, average silhouette 0.486 and total WSS 170.5; it now
  reports 49, 66.5, 0.735 and 90.7. A single cluster gives `NA`
  silhouette columns where it failed with
  `incorrect number of dimensions`.

- [`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md),
  [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  and
  [`tidy_silhouette()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette.md)
  accept factor and character cluster labels.
  [`cluster::silhouette()`](https://rdrr.io/pkg/cluster/man/silhouette.html)
  refused them with `'round' not meaningful for factors`, which turned
  away the `cluster` column of
  [`augment_kmeans()`](https://tidylearn.sheetsolved.com/reference/augment_kmeans.md)’s
  own output. Labels that read as whole numbers keep their values, so
  [`augment_dbscan()`](https://tidylearn.sheetsolved.com/reference/augment_dbscan.md)’s
  `"0"` is still noise;
  [`tidy_silhouette()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette.md)
  reports other labels as given.

- [`calc_validation_metrics()`](https://tidylearn.sheetsolved.com/reference/calc_validation_metrics.md)
  refuses `data` with no numeric column, where it summed squares over no
  columns and reported `total_wss = 0`, a perfect score:
  `calc_validation_metrics(rep(1:3, each = 50), iris["Species"])` gave 0
  where `iris[, 1:4]` gives 89.30.
  [`compare_clusterings()`](https://tidylearn.sheetsolved.com/reference/compare_clusterings.md)
  refuses the same data, where it also reported 0 with a `dist_mat`, or
  failed inside `silhouette()` with
  `NA/NaN/Inf in foreign function call` without one. Given a `dist_mat`
  and no `data`, the silhouette is still scored on its own.

- **[`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)’s
  `k_firstmax` is the first local maximum of the gap,
  `cluster::maxSE(method = "firstmax")`.** It was computed with
  [`which.max()`](https://rdrr.io/r/base/which.min.html), maxSE’s
  `"globalmax"`, so it always equalled `k_globalmax`: 8 for
  `iris[, 1:4]` with `max_k = 8`, `B = 10`, `nstart = 5` and
  `set.seed(1)`, where firstmax gives 5.
  [`print()`](https://rdrr.io/r/base/print.html) labels globalmax as the
  most liberal of the three and firstmax as the middle ground.

- `optimal_hclust_k(method = "gap")` refuses a tree built with
  `distance = "gower"` on non-numeric columns, naming them and pointing
  to `method = "silhouette"`. `clusGap()` draws its reference data
  uniformly over each numeric column, so the refit dropped the factors
  and returned a gap curve for clusterings the tree never made. A Gower
  tree on numeric columns, and a tree on any other distance, are
  unaffected.

- **[`suggest_eps()`](https://tidylearn.sheetsolved.com/reference/suggest_eps.md)
  reads the k-NN distance at `k = minPts - 1`, the neighbours a core
  point needs besides itself, as `dbscan::kNNdistplot(minPts = )`
  does.** It used `k = minPts`, the radius for one more neighbour:
  `suggest_eps(iris[, 1:4], minPts = 5)$eps` was 0.758 and is now 0.718,
  the 95th percentile of `dbscan::kNNdist(k = 4)`. `minPts` below 2 is
  an error.

- **`tidy_pca(method = "princomp", center = FALSE)` warns that
  [`princomp()`](https://rdrr.io/r/stats/princomp.html) always centres
  and records `center = TRUE` in `$settings`.** It returned centred
  scores while `$settings$center` said `FALSE`.

- [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)
  returns the dimensions classical MDS supports when `ndim` asks for
  more: [`cmdscale()`](https://rdrr.io/r/stats/cmdscale.html) keeps only
  dimensions with a positive eigenvalue, and naming `ndim` columns
  failed with `length of 'dimnames' [2] not equal to array extent`.
  `ndim` outside 1 to n - 1 is refused by name.
  [`print()`](https://rdrr.io/r/base/print.html) on a `tidy_mds` result
  counts its `Dim` columns: a fit on a tibble, which has no row labels,
  printed “Dimensions: 1” for two.

- [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  refuses a dist object with a message pointing to
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md).
  It passed the distances to
  [`cluster::clara()`](https://rdrr.io/pkg/cluster/man/clara.html),
  which samples observations and takes no distance matrix, and failed
  there with
  `'x' is a "dist" object, but should be a data matrix or frame`.

- The `cols` argument of
  [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md),
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md),
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md),
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md),
  [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md),
  [`tidy_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_dist.md)
  and
  [`tidy_pca()`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md)
  takes tidy-select expressions, as documented. Only a character vector
  worked: `cols = c(Sepal.Length, Sepal.Width)` failed with
  `object 'Sepal.Length' not found`, and
  [`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html)
  failed too.

- [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md),
  [`suggest_eps()`](https://tidylearn.sheetsolved.com/reference/suggest_eps.md),
  [`plot_knn_dist()`](https://tidylearn.sheetsolved.com/reference/plot_knn_dist.md),
  [`explore_dbscan_params()`](https://tidylearn.sheetsolved.com/reference/explore_dbscan_params.md)
  and
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  accept a numeric matrix of coordinates. They failed with
  `no applicable method for 'select'`.

- [`tidy_knn_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_knn_dist.md),
  and through it
  [`suggest_eps()`](https://tidylearn.sheetsolved.com/reference/suggest_eps.md)
  and
  [`plot_knn_dist()`](https://tidylearn.sheetsolved.com/reference/plot_knn_dist.md),
  refuse data with no numeric column, or with a missing value, in words
  that say so, as
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  does; dbscan stopped with `the provided data has 0 columns!` or
  `data/distances cannot contain NAs for kNN (with kd-tree)!`.
  [`tidy_clara()`](https://tidylearn.sheetsolved.com/reference/tidy_clara.md)
  refuses data with no numeric column the same way, where
  [`cluster::clara()`](https://rdrr.io/pkg/cluster/man/clara.html)
  reported
  `Each of the random samples contains objects between which no distance can be computed`.

- **[`standardize_data()`](https://tidylearn.sheetsolved.com/reference/standardize_data.md)
  standardises a rowwise tibble over its whole columns.**
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) works
  on a rowwise tibble one row at a time, and a single value has no
  spread, so every standardised value came back `NaN` (with
  `scale = FALSE`, every centred value was 0). The rowwise structure and
  its identifier columns are kept. A grouped tibble is still
  standardised within each group, which the documentation now says.

- [`plot_clusters()`](https://tidylearn.sheetsolved.com/reference/plot_clusters.md)
  and
  [`create_cluster_dashboard()`](https://tidylearn.sheetsolved.com/reference/create_cluster_dashboard.md)
  keep the cluster column off the default axes. As the first numeric
  column, an integer cluster column became the x axis.
  [`create_cluster_dashboard()`](https://tidylearn.sheetsolved.com/reference/create_cluster_dashboard.md)
  now skips the scatter plot when the data has fewer than two numeric
  columns besides the cluster column. With fewer than two numeric
  columns counting the cluster column it failed with “`grobs` must be a
  single grob or a list of grobs”; with an integer cluster column and
  one other numeric column it plotted that column against the cluster
  labels. It leaves out the silhouette line when the metrics have none,
  where it warned about an unknown column and printed NA, and returns
  its plots as a named list (`clusters`, `sizes`, `metrics`).

- [`plot_dendrogram()`](https://tidylearn.sheetsolved.com/reference/plot_dendrogram.md)
  draws a `tl_model(method = "hclust")` model. It passed the model to
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html), which
  reached
  [`plot.tidylearn_model()`](https://tidylearn.sheetsolved.com/reference/plot.tidylearn_model.md)
  and failed with “unused arguments (main = …, xlab = …)”. Another
  tidylearn model is refused by name.

- [`plot_knn_dist()`](https://tidylearn.sheetsolved.com/reference/plot_knn_dist.md)
  labels a percentile that is not a whole percent: `percentile = 0.975`
  failed with `invalid format '%d'`, and so did 0.57, since `0.57 * 100`
  is 56.99999999999999. It now reads “(97.5% percentile)”.

- **[`recommend_products()`](https://tidylearn.sheetsolved.com/reference/recommend_products.md)
  suggests only items the basket lacks, each once, so its
  recommendations change.** Rules whose right-hand side was already in
  the basket were returned: for the basket `whole milk`,
  `other vegetables`, `yogurt`, `root vegetables`, `tropical fruit` on
  the `Groceries` rules at `support = 0.001`, `confidence = 0.5` it
  returned `{yogurt}` and then `{other vegetables}` four times, and now
  returns no rows, since every rule that fires suggests an item the
  basket holds. A product suggested by several rules is listed once,
  with its highest-lift rule. Rules are matched on their item lists, so
  an item name containing a comma, such as `"salt, iodised"`, can fire a
  rule; splitting the label on `","` cut it in two.

- **[`filter_rules_by_item()`](https://tidylearn.sheetsolved.com/reference/filter_rules_by_item.md)
  and
  [`find_related_items()`](https://tidylearn.sheetsolved.com/reference/find_related_items.md)
  match whole items, so they return fewer rules.**
  [`grepl()`](https://rdrr.io/r/base/grep.html) on the labels matched
  `"coffee"` inside `"instant coffee"`, returning 84 of the `Groceries`
  rules where 80 contain coffee, and `"ham"` inside `"hamburger meat"`,
  207 rules where 102 contain ham; 19 `Groceries` items occur inside
  other item names. `item` must be a single name: a vector was matched
  on its first element, with a warning.

- A frequent-itemsets result from
  [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md)
  prints and inspects as itemsets.
  [`print()`](https://rdrr.io/r/base/print.html) read the rules table,
  which itemsets do not have:
  `tidy_apriori(Groceries, support = 0.05, target = "frequent itemsets")`
  printed nine min/max warnings and `Support: Inf - -Inf`, then failed
  with `no applicable method for 'slice'`, as
  [`inspect_rules()`](https://tidylearn.sheetsolved.com/reference/inspect_rules.md)
  did. The result gains `itemsets_tbl` (`itemset_id`, `itemset`, `size`,
  the quality measures and an `items` list column), which
  [`print()`](https://rdrr.io/r/base/print.html) and
  [`inspect_rules()`](https://tidylearn.sheetsolved.com/reference/inspect_rules.md)
  read;
  [`inspect_rules()`](https://tidylearn.sheetsolved.com/reference/inspect_rules.md)
  sorts itemsets by support by default, since they have no lift.

- [`summarize_rules()`](https://tidylearn.sheetsolved.com/reference/summarize_rules.md)
  refuses an itemsets result with a message saying it holds itemsets; it
  returned `Inf`, `-Inf` and `NA` summaries with nine warnings.
  [`recommend_products()`](https://tidylearn.sheetsolved.com/reference/recommend_products.md),
  [`filter_rules_by_item()`](https://tidylearn.sheetsolved.com/reference/filter_rules_by_item.md),
  [`find_related_items()`](https://tidylearn.sheetsolved.com/reference/find_related_items.md)
  and
  [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md)
  refuse it with the same message, where they failed with
  `no applicable method for 'filter' applied to an object of class "NULL"`
  or, for
  [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md),
  an arules method-dispatch error.

- **[`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md)
  plots the `top_n` rules with the highest lift, so it draws other
  rules.** It took the first `top_n` in mining order under the subtitle
  “Top 50 rules (colored by lift)”: on the `Groceries` rules at
  `support = 0.001`, `confidence = 0.5` it plotted lifts 2.04 to 16.7,
  where the 50 highest run from 8.08 to 19.0. The subtitle now reads
  “Top 50 rules by lift”. The other methods draw the same top-lift
  rules.

- **`inspect_rules(decreasing = FALSE)` returns the `n` lowest-ranked
  rules, lowest first, as arules’ `head(by = )` does.** It took the `n`
  highest and only reversed their order: lifts 16.4, 16.7 and 19.0 for
  `by = "lift", n = 3` on the `Groceries` rules, where the three lowest
  are 1.96.

#### Plots and tables

- [`plot()`](https://rdrr.io/r/graphics/plot.default.html) on a
  `"ridge"`, `"lasso"` or `"elastic_net"` model draws
  `type = "importance"` with
  [`tl_plot_importance_regularized()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_regularized.md),
  where it said the plot was not implemented. `type = "residuals"` works
  for every regression method, and `type = "diagnostics"` is refused by
  name outside `"linear"`, `"polynomial"` and `"logistic"` (the models
  fitted by [`lm()`](https://rdrr.io/r/stats/lm.html) or
  [`glm()`](https://rdrr.io/r/stats/glm.html)), where it failed inside
  [`rstandard()`](https://rdrr.io/r/stats/influence.measures.html).
  Residuals came from
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html), which is
  `NULL` on a glmnet fit, so the plot was returned and failed when
  printed; they now come from the model’s predictions, on the scale the
  formula fits. Residuals against predicted values also failed for a
  linear model whose data had a missing value, with
  `Can't recycle input of size 32 to size 31`.

- [`plot()`](https://rdrr.io/r/graphics/plot.default.html) with
  `type = "roc"`, `"precision_recall"`, `"calibration"` or `"confusion"`
  scores rows against the classes the model was trained on. The classes
  were read off the data, so a test split of
  `iris[iris$Species != "setosa", ]` still declaring setosa was refused
  as multiclass, and a test factor with its levels reordered switched
  the positive class of the precision-recall and calibration curves.
  Rows missing a response or a prediction are left out with a warning
  giving the count: ROC and precision-recall stopped with ROCR’s
  `'predictions' contains NA`, and the confusion counts summed to 30 of
  32 with no message. ROC and precision-recall need rows of both classes
  and now say so, where ROCR reported
  `Number of classes is not equal to 2`.

- `plot(type = "actual_predicted")` compares on the scale the model was
  fitted on and leaves incomplete rows out of its statistics, with a
  warning giving the count. For `log(mpg) ~ wt` it plotted raw `mpg`
  against predictions of `log(mpg)` and reported a correlation of 0.868,
  where the fitted scale gives 0.893, and one missing value made the
  subtitle `Correlation: NA, R-squared: NA`.

- [`tl_plot_intervals()`](https://tidylearn.sheetsolved.com/reference/tl_plot_intervals.md)
  works with `y ~ .`, which failed with `argument 1 is not a vector`,
  and plots the observed response on the scale of the bands: for
  `log(mpg) ~ wt` it drew raw `mpg` against bands for `log(mpg)`. A
  model without an [`lm()`](https://rdrr.io/r/stats/lm.html) fit is
  refused by name, where `"ridge"` failed asking for `newx`. Data
  without the response gets the bands alone, and a row with a missing
  predictor is left out with one warning, where ggplot2 gave four.

- **Importance for `"ridge"`, `"lasso"` and `"elastic_net"` no longer
  depends on the units of each predictor, so its values change in
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  [`tl_plot_importance_regularized()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_regularized.md)
  and
  [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md).**
  It was the absolute coefficient, so rescaling `hp` to hundreds
  multiplied its importance about a hundredfold without changing a
  prediction. Each coefficient is now multiplied by the standard
  deviation of its design column over the rows the model was fitted on.
  A multiclass model, which failed with
  `non-numeric argument to mathematical function`, now takes each
  predictor’s largest value across classes. A penalty that drops every
  predictor is reported as such, where it gave an empty table and a
  [`max()`](https://rdrr.io/r/base/Extremes.html) warning.

- **Importance for a random forest whose permutation importance is
  negative for every feature keeps randomForest’s order.**
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  `plot(model, type = "importance")` and the importance comparison
  divided by the largest value, itself negative, so %IncMSE of -6.20,
  -0.87 and -6.61 (`set.seed(7)`, three noise predictors) became 709,
  100 and 756 and the least useful feature ranked first. With no
  positive value the largest magnitude now sets the scale: -93.8, -13.2
  and -100. Any set with a positive value is scaled as before.

- Variable importance works for more models. A forest fitted with
  `importance = FALSE` failed with `subscript out of bounds`; it now
  uses the impurity measure the forest does have.
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  which documented xgboost support, refused xgboost models; they now
  report gain. `plot(model, type = "importance")` shares the table’s
  extraction, and the dashboard’s importance panel, which showed an
  error for `"ridge"`, `"lasso"` and `"elastic_net"`, now plots them.
  [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  with no supported model failed inside dplyr; it now says so.

- Importance for a tree with no splits stops with “No feature has
  non-zero importance: the tree has no splits”, in
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  `plot(model, type = "importance")` and the dashboard’s importance
  panel. They failed with
  `` Column `importance` not found in `.data` ``, and so did any
  [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  that included the tree, which now draws with that tree at zero.

- [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  counts as zero a feature a model was given but did not use. The
  ranking averaged each feature over only the models that kept it, so a
  feature a lasso dropped could outrank one both models used. A tree,
  forest or boost model is given the variables of an interaction, never
  the interaction itself, so `wt:hp` gets no bar from it.

- **Importance for `method = "boost"` names the columns gbm fitted on,
  so the rows it reports change.** gbm names its relative influence
  after the formula’s terms but computes it over the variables they use,
  so for `mpg ~ wt * hp + qsec`
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md),
  `plot(model, type = "importance")` and the importance comparison
  listed a `wt:hp` at zero, a column gbm never had, and for
  `mpg ~ wt:hp + qsec` they failed with gbm’s “row names contain missing
  values”. Both now report `wt`, `hp` and `qsec`, each with its own
  influence.

- [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  names a factor predictor once for every model. randomForest, rpart and
  gbm report the factor (`Species`) and glmnet and xgboost its design
  columns (`Speciesversicolor`, `Speciesvirginica`), so a forest and a
  lasso of `Sepal.Length ~ .` on iris (`set.seed(1)`) drew 9 bars over 6
  features: the one predictor appeared three times, each with one
  model’s bar. Design columns now map back to their formula term, which
  takes its largest column, and the same pair draws 8 bars over 4
  features. A non-syntactic name was split the same way: rpart reports
  `car weight` and glmnet `` `car weight` ``, so a lasso and a tree
  showed two features; they now show one, named without backquotes. A
  lasso that penalised every predictor away dropped out of the plot with
  only a [`max()`](https://rdrr.io/r/base/Extremes.html) warning; it now
  stays at zero, with a warning naming it.

- [`tl_plot_importance_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_importance_comparison.md)
  refuses repeated `names`. `names = c("A", "A")` drew both models’ bars
  in the same places; it is now an error, as it already was for
  [`tl_plot_model_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_model_comparison.md)
  and
  [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md).

- An unrecognised `lambda` is now rejected by name. Anything that was
  not `"1se"` or `"min"` used to reach `glmnet::coef()` as a penalty, so
  a typo such as `"1SE"` failed with
  `non-numeric argument to binary operator` and a vector of penalties
  with `the condition has length > 1`, neither of which names the
  argument at fault. A numeric `lambda` outside the penalties the model
  was fitted over is refused as well. glmnet returns the coefficients at
  the nearest end of the path for it, which the table labelled with the
  penalty asked for: `lambda = 0` returned the coefficients at the
  smallest penalty on the path, labelled as unpenalised. The same
  penalty checks apply to regularised importance.
  [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
  takes no `...`, so a misspelt argument such as broom’s
  `conf.int = TRUE` is an error rather than a table without an interval.

- [`tl_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_coefficients.md)
  and
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  report a term the fit could not estimate — one of two collinear
  predictors, or an interaction of factors with a combination no row has
  — as a row with an `NA` estimate.
  [`summary()`](https://rdrr.io/r/base/summary.html) drops aliased terms
  from its coefficient matrix, so
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  omitted them silently: `mpg ~ wt + wt_doubled` produced a two-row
  table for a three-term formula, with nothing to show the third term
  had ever been there. A fit of full rank is unaffected.

- [`tl_plot_model_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_model_comparison.md)
  and
  [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)
  keep two models of the same method apart. Both got the same default
  name (`"linear (reg)"` in the table), so the plot drew one bar over
  the other and the table pivoted the pair into list cells. Repeated
  default names are now numbered, and
  [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)
  checks `names` has one unique entry per model, as the plot already
  checked its length.

- [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)
  and
  [`tl_plot_model_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_plot_model_comparison.md)
  with no `new_data` score the models on their training data only when
  every model was fitted on the same data. Both scored every model on
  the first model’s training rows: two linear models fitted on
  `mtcars[1:20, ]` and `mtcars[13:32, ]` were compared on the first
  one’s rows, which the second had partly never seen, without a message.
  Models fitted on different data are now an error naming them and
  asking for `new_data`. Models fitted on one data frame with different
  formulas still share it. A
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  PCA or cluster candidate stores its engineered features in place of
  the rows it was fitted on, so with a PCA candidate listed first the
  first model’s stored data was the PCA scores, and the comparison
  failed inside the candidate with “PCA was fitted on 4 column(s) … but
  new_data is missing”. Every model is now scored on the raw training
  rows of the first model that stores them, a cluster candidate
  included, and a PCA candidate through its own projection, as it
  already was when listed after one. Models fitted on PCA scores alone
  store no raw rows, and still need `new_data`.

- **[`tl_plot_lift()`](https://tidylearn.sheetsolved.com/reference/tl_plot_lift.md)
  and
  [`tl_plot_gain()`](https://tidylearn.sheetsolved.com/reference/tl_plot_gain.md)
  no longer depend on row order, so their curves change.** Rows with
  tied probabilities kept the order they arrived in, and a tree scores
  many rows alike: for a tree of `Species ~ Sepal.Width` on two iris
  classes, reversing the rows moved the gain at 10% of the population
  from 0% to 20% of responders. Each row now counts the response rate of
  its tie group, the value any tie-breaking gives on average. A row
  missing its response turned every point to `NA` and left the chart
  empty; those rows are now left out, with a warning. The bins also
  match `bins`: sizing them by rounding up gave 32 rows in 10 bins as 8,
  and a `bins` that is not a whole number of at least 1 is refused.

- **[`tl_plot_lift()`](https://tidylearn.sheetsolved.com/reference/tl_plot_lift.md),
  [`tl_plot_gain()`](https://tidylearn.sheetsolved.com/reference/tl_plot_gain.md)
  and
  [`tl_table_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_table_confusion.md)
  read the observed classes against the model’s.** The positive class
  came from the scored data’s level order, so reordering `am`’s levels
  to `c("1", "0")` made the gain chart of `am ~ wt` rank mtcars by P(am
  =

  0.  and count the 19 automatic cars as responders, where the model’s
      positive class is the 13 manual ones. Lift and gain decide binary
      from the model: a test split of `iris[iris$Species != "setosa", ]`
      still declares setosa, so both charts called a binary model
      multiclass and refused it; they now draw it. The confusion matrix
      of a two-class model gained a row of zeros for a class the scored
      data declared but did not hold, and a row for each class the model
      was never trained on. It now has one row and one column per class
      the model was trained on. Rows of any other class are left out
      with a warning, where lift and gain failed. Lift and gain on
      scored rows with no row of the positive class are an error naming
      the class, where every cumulative value was 0 / 0. A `new_data`
      without the response column is reported by name (“Response
      variable ‘am’ not found in the evaluation data”) where the charts
      said the model was not binary and the confusion table failed with
      “all arguments must have the same length”.

- [`tl_table_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_table_confusion.md)
  warns when rows are missing a response or a prediction.
  [`table()`](https://rdrr.io/r/base/table.html) dropped them, and the
  counts summed to fewer rows than were passed in without saying so.

- [`tl_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_dashboard.md)’s
  regression panels work for every method. The Residuals panel passed
  the evaluation data to
  [`tl_plot_residuals()`](https://tidylearn.sheetsolved.com/reference/tl_plot_residuals.md)
  as its plot type and failed with “the condition has length \> 1” for
  linear, lasso and forest models; it now plots residuals of the
  dashboard’s data against predictions. The Diagnostics panel printed
  the four diagnostic plots in turn and showed only the last, Residuals
  vs Leverage; it now arranges all four (with gridExtra installed). For
  a model that is not linear or polynomial it failed inside
  [`rstandard()`](https://rdrr.io/r/stats/influence.measures.html), and
  now says the plots need a linear or polynomial model. The Predictions
  table and the Residuals panel read the response as the formula’s
  left-hand side: a `log(mpg) ~ wt + hp` model listed mpg beside
  log-scale predictions, with residuals of one minus the other.

- `tl_plot_cv_results(metrics = )` draws only the metrics asked for. The
  mean lines were not filtered, so `metrics = "rmse"` still laid out a
  panel for every metric the folds scored, with fold lines in rmse
  alone. A requested metric the folds did not score is named in a
  warning, and asking only for absent metrics is an error listing the
  ones available.

- The source notes of the `tl_table*()` tables count the rows the table
  describes. Every note reported `nrow(model$data)`: coefficients for
  `Ozone ~ Temp + Wind` on airquality said n = 153 where
  [`lm()`](https://rdrr.io/r/stats/lm.html) used 116, and metrics and
  confusion tables scored on a 10-row test set said n = 22, the training
  rows. Coefficient and importance tables now count the rows the fit
  used, and metrics, confusion and comparison tables the rows scored; a
  comparison whose models scored different rows gives each model’s
  count. An xgboost fit with `weights` counts only the rows with a
  weight, which are the rows xgboost trains on.

- [`tl_table_clusters()`](https://tidylearn.sheetsolved.com/reference/tl_table_clusters.md)
  summarises only the columns an hclust or dbscan fit used. A dbscan
  model of `~ Sepal.Length + Sepal.Width` on `iris[, 1:4]` tabulated the
  means of all four measurements, while the kmeans table already showed
  only the fitted two. The cluster table no longer counts dbscan’s noise
  points as a cluster, and now averages hclust and dbscan tables around
  missing values. A long formula no longer splits a table’s source note
  in two, and
  [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  warns about arguments it does not use.

- [`tl_plot_regularization_path()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_path.md)
  draws a multiclass model, one panel per class; it failed with
  `Tibble columns must have compatible sizes`. Its feature labels were
  drawn on top of the paths they name, in the same colour, so they were
  unreadable, and two coefficients that end close together printed one
  label over the other. On `mtcars` that hid `drat` behind `qsec` and
  clipped `am` and `wt` at the panel edge. Labels now sit clear of the
  leftmost point, spread far enough apart to read, each with a leader
  line back to its own path. The room for labels grows with the longest
  name, so `Speciesversicolor` is no longer cut off at the panel edge at
  7 x 5 inches. When every path is labelled – five or fewer predictors
  at the default `label_n` – they were all drawn grey, thin and faint,
  because the colour, width and transparency scales matched their values
  by position; they are now drawn in the accent colour.

- [`tl_plot_regularization_path()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_path.md)
  refuses a model fitted at a single `lambda`. Each term was one point,
  so the plot drew no line, under a subtitle naming a `lambda.min` and
  `lambda.1se` no cross-validation had chosen. A model fitted along
  several penalties without cross-validation, as earlier versions fitted
  a `lambda` sequence, is drawn without the `lambda.min` and
  `lambda.1se` lines and says so.

- [`tl_plot_regularization_cv()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_cv.md)
  labels its axis with the measure cross-validation used. It read
  “Binomial Deviance” for every classifier, a multinomial one included,
  and “Mean Squared Error” for every regression, whatever `type.measure`
  chose.

- [`tl_plot_xgboost_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_tree.md)
  draws the tree `tree_index` names. On xgboost 3.x it drew the first
  tree whatever `tree_index` was, and warned that `tree_index` will
  become an error. A `tree_index` past the model’s last tree is refused
  with the number of trees.

- **[`tl_xgboost_shap()`](https://tidylearn.sheetsolved.com/reference/tl_xgboost_shap.md)
  reports a multiclass model’s SHAP values per class: one block of rows
  per class, with a `class` column.** It labelled a flattened slice of
  xgboost 3.x’s row x class x feature array, so on `iris` the column
  called `Petal.Length` held `Sepal.Length`’s values for virginica, and
  11 of the 18 columns were named `NA`, `NA.1` and so on. It also scores
  data without the response column, which failed with
  `object 'mpg' not found`, and applies `trees_idx` as a run of boosting
  rounds, where it was dropped.

- [`tl_plot_xgboost_shap_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_dependence.md)
  takes its SHAP values and its feature values from the same rows. It
  drew two independent samples, so for `y = 10x` over 400 rows the
  plotted correlation between the feature and its SHAP value was near
  zero. Both SHAP plots draw a multiclass model one panel per class.

- [`tl_plot_svm_boundary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_svm_boundary.md)
  takes its default axes from the model’s numeric predictors, and fills
  in only an axis that was not named. They were the first two numeric
  columns of the data, so `Species ~ Petal.Length + Petal.Width` was
  drawn over the sepal columns the model never used. The 0.5 probability
  contour of a two-class model is drawn; a check against the wrong
  fields of the e1071 fit kept it from ever appearing.

- [`tl_plot_partial_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_partial_dependence.md)
  draws the mean prediction at each grid value for a regression model.
  It took [`mean()`](https://rdrr.io/r/base/mean.html) of the tibble
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns, so every
  regression curve, the function’s own example included, was `NA`, with
  a warning for each grid value.

- [`tl_plot_deep_architecture()`](https://tidylearn.sheetsolved.com/reference/tl_plot_deep_architecture.md)
  draws through the
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) method keras
  provides for its models. It called `keras::plot_model()`, which keras
  2.x does not have, so every call failed with
  `object 'plot_model' not found`.

#### Data ingestion

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  names the cause for sources it cannot read. A web URL other than
  GitHub or Kaggle was passed to the CSV reader and reported as “File
  not found”; it is refused with a message to download the file first,
  with or without `format`. Hosts are matched on the URL’s host name, so
  `https://mirror.example.org/github.com/data.csv` is no longer taken
  for GitHub. `s3://bucket/archive.zip` was looked for as a local zip
  file, and
  [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  and
  [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
  now say they cannot read zip archives. An `NA` or empty `source` is
  refused by name; it failed with “missing value where TRUE/FALSE
  needed”. URL schemes match in any case, so `S3://` reaches the S3
  reader.

- [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)
  reads files only. Without `recursive = TRUE`,
  [`list.files()`](https://rdrr.io/r/base/list.files.html) also returns
  folders, so a folder whose name matched, such as `old.csv`, was read
  as a directory and its rows were added to the result.

- **`tl_read_zip(file = )` reads the member it was asked for.**
  `file = "train.csv"` read `full_train.csv` when the archive held both,
  because the name was matched as part of each member’s base name and
  the first match in file order won, with no message under
  `.quiet = TRUE`. An exact path within the archive now wins, then an
  exact file name, then part of a path, and a name that matches several
  members at the deciding step is an error listing them. A member in a
  folder can be named by its path, such as `"2024/sales.csv"`; of two
  members called `sales.csv`, only the first could be read before.

- [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
  and
  [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  refuse an archive with a member whose name is an absolute path such as
  `/x.csv`, has a `..` component, or, on Windows, names a drive
  (`C:x.csv`). Before R 4.5.1, which tidylearn supports,
  [`unzip()`](https://rdrr.io/r/utils/unzip.html) extracts such members
  as written, so a crafted archive could plant a file anywhere the user
  can write, such as an `.Rprofile` that runs at the next R start. From
  R 4.5.1 [`unzip()`](https://rdrr.io/r/utils/unzip.html) drops the `..`
  components instead, so on those versions such an archive used to be
  read. Every `..` component is refused, including one that stays
  inside, such as `a/../b.csv`.

- [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
  reads file links from `www.github.com`, links carrying a query such as
  `?raw=true`, and links using `/raw/` in place of `/blob/`. The
  raw-file address was built by substituting text: `?raw=true` hid the
  file extension, `www.github.com` was taken for “owner/repo” shorthand,
  and in a repository owned by an account called `blob` the owner’s name
  was removed in place of the `/blob/` segment. A link to something
  other than a file, such as a `/tree/` folder, is refused by name, and
  a `raw.githubusercontent.com` link keeps its query, where a private
  file’s token lives.

- [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  reads the dataset a Kaggle URL names, whichever tab it was copied
  from. The slug was the URL’s last two path segments, so
  `https://www.kaggle.com/datasets/uciml/iris/data` downloaded the
  dataset `iris/data`, another owner’s, which passed validation; `/code`
  and `/versions/2` did the same, and a trailing slash or a `?select=`
  query was refused with a message about something else. A competition
  URL is read as a competition without `type = "competition"`, and
  contradicting it with `type` is an error. `type` itself is checked: a
  misspelt value was treated as `"dataset"`.

- `tl_read_kaggle(dest = )` downloads into a directory of its own and
  copies the download into `dest`. It downloaded into `dest` and
  unpacked every zip there, so an unrelated archive in the folder
  overwrote the user’s files, and a later call could return an earlier
  download’s files, re-extracted and so the newest, in place of the
  dataset asked for. Files from this download still replace files of the
  same name in `dest`; nothing else there is read, unpacked or changed.

- [`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md)
  connects with a `postgres://` connection string. It passed the string
  to RPostgres as `dsn`, which libpq has no keyword for, so every
  connection string failed with ‘invalid connection option “dsn”’. The
  URL is taken apart into host, port, user, password and database name;
  query parameters such as `sslmode=require` go to libpq as connection
  keywords; and the named arguments fill parts the URL leaves out, which
  keeps a password out of it.
  [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)
  uses its `dbname`, `user`, `password` and `port` arguments for the
  parts a `mysql://` URL leaves out. With a URL they were ignored, so
  `port = 3307` alongside `"mysql://host/db"` connected to 3306.

- Database connection strings are percent-decoded, so a password written
  `p%40ss` reaches the server as `p@ss`; it was sent encoded.
  [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)
  refuses query parameters: `?ssl-mode=REQUIRED` became part of the
  database name, and RMariaDB ignores arguments it does not know, so
  passing it on would connect without the TLS asked for. Pass
  `RMariaDB::dbConnect()` arguments such as `ssl.ca` through `...`
  instead. The progress message from
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  and the `tl_source` attribute redact a password given as `?password=`,
  as a libpq keyword, after an empty user name
  (`postgres://:secret@host`) or under an upper-case scheme, all of
  which were shown in the clear. The same goes for the SSL key
  passphrase (`?sslpassword=`), for the other secret-bearing keywords
  newer libpq versions take (`oauth_client_secret`, `scram_client_key`,
  `scram_server_key`), and for a password written in quotes
  (`password='two words'`) or with spaces around `=`. A local path is
  left alone: the redaction took `C:\Users\ana@corp\data.csv` for
  `user:password@host` and printed `C:***@corp\data.csv`. The “File not
  found” error redacts the path it names, which printed a connection
  string’s password in full.

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  refuses a URL whose scheme it has no reader for, such as `ftp://`,
  with a message naming the scheme and listing the sources it reads; it
  reported `File not found`. A `file://` URL is read as the local path
  it names, percent-decoded, where it was reported as not found, and a
  `file://` URL naming another machine is refused. A source with a
  scheme can no longer be sent to a file reader with `format`:
  `tl_read("postgres://ana:secret@host/db", format = "csv")` failed with
  “File not found” and printed the password.
  `bigquery://project/dataset` sources are detected and read with
  [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md),
  where they stopped with “Cannot detect format”.

- [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md),
  [`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md)
  and
  [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)
  check `project`, `dsn` and `query` before contacting a server, and the
  message names the argument. A `NULL` `project` or `dsn` failed with
  “argument is of length zero”, and an `NA` `dsn` or `query` was handed
  to the database driver.
  [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md)
  also refuses a `bigquery://` URI with an empty project or dataset,
  such as `bigquery:///my_dataset`.

- `tl_read_bigquery(dataset = )` reaches the query as its default
  dataset, so unqualified table names resolve against it, as documented.
  It only labelled the result.

#### Compute and cloud

- [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md),
  and with it `tl_model(compute = "auto")`, handle problems past 2^31 -
  1 cells. Rows times predictors was an integer product, so 1e7 rows by
  250 predictors overflowed to `NA` and both failed with “missing value
  where TRUE/FALSE needed”, on the inputs the advisor exists to size.

- **[`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  counts predictors from the terms the formula expands to against the
  data, so its estimates change.** It counted the variable names on the
  right-hand side, so on a frame of 200 predictors plus `id` and `y`,
  `y ~ . - id` was sized as 2 predictors and `y ~ 1` as 201. A fitted
  model is sized from its own formula the same way.

- **[`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  sizes the fit tidylearn would run, so its estimates change.** A
  runtime hyperparameter left out takes the default of the method’s fit
  function. It assumed 10 epochs of 128 units for `"deep"`, where
  [`tl_fit_deep()`](https://tidylearn.sheetsolved.com/reference/tl_fit_deep.md)
  trains 30 epochs through layers of 32 and 16, and a hidden layer of 10
  for `"nn"` against
  [`tl_fit_nn()`](https://tidylearn.sheetsolved.com/reference/tl_fit_nn.md)’s
  5, which halves the `"nn"` estimate. On iris’s four predictors the
  `"deep"` estimate is 3.8 times the old one. `"deep"` reads
  `hidden_layers`; `units`, which the help page listed, is no tidylearn
  argument and is ignored. A hyperparameter the estimate reads must be a
  positive number: `nrounds = NA` failed with “attempt to select less
  than one element”, and a negative value gave a negative runtime.

- [`print()`](https://rdrr.io/r/base/print.html) on a
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  result, and its reasoning and notes, write large numbers out in full
  with thousands separators. A round value printed in scientific
  notation: a 100,000 MB peak read “1e+05”.

- [`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)
  drops the trailing dot of a fully qualified name.
  `"fits.example.com."` was stored with the dot, which no endpoint’s
  host carries, so the addition matched nothing.

### Documentation

- [`?tl_model`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  lists the arguments tidylearn takes for each method – among them
  `degree` for `"polynomial"`, `alpha`, `lambda` and `cv_folds` for the
  regularised methods, which were documented only on internal pages, and
  `mds_method` and `k` (or `ndim`) for choosing an MDS variant and its
  dimensions – names `"polynomial"` among the methods, and says that a
  logical response is a regression for every method but `"logistic"`.

- The reporting vignette said every plot function returns a ggplot2
  object.
  [`plot_dendrogram()`](https://tidylearn.sheetsolved.com/reference/plot_dendrogram.md),
  which [`plot()`](https://rdrr.io/r/graphics/plot.default.html) uses
  for an hclust model,
  [`tl_plot_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tree.md)
  and
  [`tl_plot_nn_architecture()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_architecture.md)
  draw with base graphics,
  [`tl_plot_xgboost_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_tree.md)
  returns a DiagrammeR widget, and the dashboards and
  `plot(type = "diagnostics")` return a grid or a list of plots; the
  vignette now lists the exceptions. It also said its reporting code
  runs unchanged with `method = "svm"`, for which
  [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md)
  errors. The README claimed ggplot2 output regardless of model type and
  labelled `plot(linear_model)` as diagnostic plots, where it draws
  actual against predicted values. The same “tibble or ggplot2 object”
  claim, in the README’s Philosophy section and the getting-started
  vignette, now covers predictions, metrics and most plots, and
  getting-started counts 20 methods from 13 packages where it said 20
  packages.

- The supervised-learning vignette counted eleven supervised methods
  where there are thirteen, and said regularised regression and SVM need
  pre-scaled inputs. glmnet and e1071 standardise internally by default,
  so scaling beforehand leaves their predictions unchanged. The advice
  now covers `"nn"` alone.

- The unsupervised-learning vignette called “Cluster 1” the clean
  k-means cluster, but k-means numbers clusters arbitrarily and the
  vignette set no seed, so *setosa* came out as cluster 1, 2 or 3
  depending on the random state. The vignette now sets a seed and names
  the *setosa* cluster. It also said single linkage puts almost
  everything in one cluster, beside output showing 98 of 150 rows; it
  now gives that count.

- The AutoML vignette said a 30-second budget at the default five folds
  skips cross-validation entirely; on iris every model in that run is
  cross-validated. Its time-budget section now lists the gates the code
  applies: the forest and the advanced models need `time_budget >= 30`,
  the PCA and cluster phases start only while more than 5 seconds and
  10% of the budget remain, the advanced phase only while more than 40%
  remains, and a model is cross-validated only while more than 30%
  remains. The `glm.fit` convergence warnings it hides, counted as a
  dozen, came to six to eight in four runs and are now described as
  several. The vignette now sets a seed, so its folds and leaderboards
  repeat between builds.

- [`?tl_auto_ml`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  describes the time budget as the code applies it. It said a budget
  under 30 seconds skips cross-validation and fits two models; its own
  example, `time_budget = 10` on iris, fits one tree and cross-validates
  it.
  [`?tl_get_best_model`](https://tidylearn.sheetsolved.com/reference/tl_get_best_model.md)
  says the returned model expects preprocessed data and is meant to be
  used through
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md);
  [`predict()`](https://rdrr.io/r/stats/predict.html) on it with raw
  rows returns wrong values without a warning. The help page for
  [`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)
  had the title “Integration Functions: Combining Supervised and
  Unsupervised Learning”; it now has its own.

- The help pages of the twelve readers of local files –
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md),
  [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md),
  [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md),
  [`tl_read_csv()`](https://tidylearn.sheetsolved.com/reference/tl_read_csv.md),
  [`tl_read_tsv()`](https://tidylearn.sheetsolved.com/reference/tl_read_tsv.md),
  [`tl_read_excel()`](https://tidylearn.sheetsolved.com/reference/tl_read_excel.md),
  [`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md),
  [`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md),
  [`tl_read_parquet()`](https://tidylearn.sheetsolved.com/reference/tl_read_parquet.md),
  [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md),
  [`tl_read_db()`](https://tidylearn.sheetsolved.com/reference/tl_read_db.md)
  and
  [`tl_read_sqlite()`](https://tidylearn.sheetsolved.com/reference/tl_read_sqlite.md)
  – have examples that run, on temporary files or on the sample files
  readr and readxl ship, each guarded on the suggested package it needs.
  Every line of their examples was commented out. The six readers that
  need a server, credentials or the network have their examples as code
  in `\dontrun{}`. The
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  example fits xgboost on two threads.

- The `@return` sections of
  [`tidy_pam()`](https://tidylearn.sheetsolved.com/reference/tidy_pam.md),
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md),
  [`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md),
  [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md)
  and
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)
  name the elements the functions return. They listed `silhouette` (the
  elements are `silhouette_avg` and `silhouette_data`), `core_points`
  (the flags are `clusters$is_core`) and the class `"tidy_rules"` (it is
  `"tidy_apriori"`), and left out
  [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md)’s
  `sizes`,
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md)’s
  `distance_method`, and
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)’s
  `summary`, `eps` and `minPts`.

- [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md)’s
  `@return` said only `method = "scatter"` returns a ggplot and that
  other methods draw as a side effect. arulesViz returns ggplot objects
  for `"graph"`, `"grouped"` and `"matrix"` too; only `"paracoord"`,
  which draws with grid, returns a grid `vpPath`. Its `rules_obj`
  documentation no longer lists a table of rules, which the function
  refuses.

- `inst/examples/unified_workflow.R` printed “Transfer learning model
  built on over 3 principal components”: it read `$spec$method`, which a
  [`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)
  result does not have. It also printed “Original data: 5 features with
  missing values” when one of the five had any. The lines now read
  “Transfer learning model: forest fitted on 3 principal components” and
  “Original data: 5 features, 1 with missing values”, and the
  processed-data line counts its missing values the same way.
  `tests/testthat/test-examples.R` checks all three.

- `PACKAGE_ARCHITECTURE.md`’s four links into the README were 404s on
  the documentation site. pkgdown rewrote `README.md#...` to
  `README.html`, a page it never builds, because it publishes the README
  as the site’s home page. They now point at the home page’s sections
  directly, which also works when the file is read on GitHub.

- The hex logo has sharp corners, in line with other R package hex
  stickers. The artwork is otherwise unchanged.

## tidylearn 0.5.0

CRAN release: 2026-09-02

### New Features

#### Cloud compute (security guards)

- [`tl_cloud_consent()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_consent.md)
  — grants or revokes permission for the rest of the R session to upload
  training data to your Modal account. Cloud fits otherwise require
  `confirm_upload = TRUE` on every call. The lock is never written to
  disk and does not survive an R restart, and tidylearn never prompts
  interactively, so scripts and CI behave the same as an interactive
  session.

- Cloud endpoints are read from the `TIDYLEARN_MODAL_ENDPOINT`
  environment variable and validated before any request is built: the
  scheme must be `https` and the host must be on the allowlist.
  Lookalikes such as `modal.run.example.com` or `evil-modal.run` are
  rejected. The endpoint is user-supplied configuration, so this check
  is what stops a typo or a modified variable sending training data
  somewhere other than Modal. An environment variable is used rather
  than an R option because an option can be set silently by a shared
  `.Rprofile`.

- [`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)
  and
  [`tl_cloud_allowed_hosts()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allowed_hosts.md)
  — the allowlist defaults to Modal’s own domains, and Modal customers
  serving Web Functions from a custom domain can extend it. Extension is
  a per-session call rather than an option or environment variable, for
  the same reason: nothing inherited from the environment should be able
  to add an upload destination. Added hosts must be bare host names, and
  a single label such as `"com"` is refused because it would open an
  entire top-level domain.

  These implement T2 and T9 of
  `system.file("security/threat-model.md", package = "tidylearn")`.
  Submission itself is still not wired up — `compute = "cloud"`
  continues to error.

#### Cloud compute (model serialisation)

- Internal helpers now convert a fitted model to bytes and back for
  transport from a remote worker. Twelve of the thirteen supervised
  methods survive base R serialisation unchanged, xgboost included — its
  booster is embedded in the byte stream rather than left as a dangling
  pointer.

  `method = "deep"` is the exception and is handled separately: a keras
  model is a reference to a Python object and cannot cross a process
  boundary that way, so its weights travel as their own hdf5 payload via
  [`keras::serialize_model()`](https://rdrr.io/pkg/keras/man/serialize_model.html).
  Detection is by the presence of a Python object rather than by method
  name or keras class, because keras renamed its classes between
  versions and matching those would silently stop detecting models on
  one side of the change.

### Bug Fixes

Several of these changed reported numbers. Results produced by 0.4.0 and
earlier should be recomputed.

#### Degenerate inputs

- `tl_model(method = "forest")` hung indefinitely on a classification
  response whose predictors were all constant. randomForest’s
  classification path keeps drawing `mtry` candidates looking for a
  split that cannot exist, and the loop is C-level, so it ignored
  interrupts and the session had to be killed. It was reachable through
  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md),
  whose default candidates include a forest, and through
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md),
  whose baselines do. Now refused before the call, naming the columns.
  Regression is unaffected and still fits, as does a frame where only
  some predictors are constant. The predictor set is read through
  [`terms()`](https://rdrr.io/r/stats/terms.html), so a `.` is expanded
  against the data and an exclusion such as `y ~ . - id` is honoured.

- A character specification such as `"Species ~ ."` was read as a
  regression problem by every entry point except
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md).
  `tl_model_supervised()` coerces with
  [`as.formula()`](https://rdrr.io/r/stats/formula.html), but that
  happens after
  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md),
  [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md),
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md),
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  and the two tuners have already called `all.vars(formula)[1]` – and
  [`all.vars()`](https://rdrr.io/r/base/allnames.html) on a string is
  `character(0)`, so the response name came back `NA` and `data[[NA]]`
  was `NULL`. `tl_auto_ml(iris, "Species ~ .")` announced “task:
  regression” and returned an unranked leaderboard;
  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  scored a classification tree with `rmse` and returned a pipeline whose
  every metric was `NA`, warning only that the values were missing.
  Coercion now happens at each entry point, before anything reads the
  formula, and an argument that is neither a formula nor a string that
  parses as one is refused by name.

- A repeated name in `tl_pipeline(models = ...)` silently discarded a
  model. The training loop indexes `models[[model_name]]`, which
  resolves to the first match, so `list(a = tree, a = forest)` fitted
  the tree twice and never fitted the forest. Repeated names are now
  refused, next to the existing guard for unnamed ones.

- Malformed entries in `models` reported base R internals that named
  neither the model nor the mistake: a spec with no `method` gave
  “missing value where TRUE/FALSE needed”, a spec that was not a list
  gave “\$ operator is invalid for atomic vectors”, a two-element
  `method` gave “‘length = 2’ in coercion to ‘logical(1)’”, and an
  unsupervised method gave “undefined columns selected”. Each is now
  checked before the run starts and names the offending model.

- `evaluation$cv_folds` and `evaluation$train_prop` were unvalidated.
  `train_prop = 0` reached base R as “result would be too long a
  vector”, `train_prop = 1` surfaced as a ROCR complaint about class
  counts, and `train_prop = 1.5` as “cannot take a sample larger than
  the population”; bad fold counts arrived as rsample errors naming `v`,
  which is not an argument of anything the caller wrote. Both are now
  checked against their own names, and both are checked again against
  the row count when the pipeline runs: more folds than rows is reported
  as such, and so is a `train_prop` that is in range but rounds to an
  empty side on a small frame. `evaluation$validation` and
  `evaluation$best_metric` were in the same position one step earlier –
  set to `NULL` they were dropped from the list and reached `%in%` as
  “argument is of length zero” – and are now reported by name too.

- An unrecognised name in `evaluation$metrics` was accepted, computed
  nothing, and left the run warning that all values were `NA` – the
  symptom rather than the cause. Unknown metrics are now refused with
  the list of available ones, matching what
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  already did. An empty metric set is refused too; it previously
  rendered the `best_metric` error as a bare full stop. The list is
  judged against the task the pipeline will actually fit,
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)’s
  logistic rule included: `method = "logistic"` on a 0/1 integer
  response fits a classification model and each fold reports
  classification metrics, so `accuracy` and `auc` are accepted there.

- A numeric response with both logistic and any other supervised method
  among the candidates gave one run two tasks. The leaderboard holds one
  set of metrics, so whichever way they were chosen the other models
  scored `NA` and dropped out of the comparison silently – `logistic`
  plus `linear` on a 0/1 column ranked logistic at 0.6998 and reported
  nothing at all for linear. The mixture is now refused where the models
  are read, naming the methods on each side. A factor response is
  unaffected: there is one task there, and mixing methods is the
  ordinary case.

- A response that is not a column of `data` – a typo in the formula –
  read as `NULL` in
  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md),
  which set regression defaults and failed several steps later inside
  rpart with “object ‘Speces’ not found”. It is now refused where the
  formula is read, listing the columns that are there.

- An intercept-only formula (`y ~ 1`) reached
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  and failed with “result would be too long a vector”. A pipeline
  preprocesses and scores predictors, so it now says it needs at least
  one.

- A single-class response was named plainly only by logistic regression.
  Every other classification method reported whatever its backend hit
  first: rpart “number of rows of matrices must match (see arg 2)”,
  glmnet “non-conformable arguments”, e1071 “Model is empty!”, xgboost a
  complaint about `num_class`. Ten of the thirteen supervised methods
  now give the same message, naming the response and the class it holds.
  `linear` and `polynomial` keep the numeric-response message they
  already had, which now says the response holds a single class rather
  than offering classification methods that would refuse it in turn, and
  logistic regression keeps its own wording.

- `method = "forest"` and `method = "svm"` derived their defaults from
  the number of columns in the frame rather than the number of
  predictors in the formula, so an explicit formula over a wider frame
  got the wrong one. `mpg ~ wt + hp` on `mtcars` asked randomForest for
  `mtry = 3` of 2 predictors, which it reset with a warning, and asked
  e1071 for a kernel width of 1/10 instead of 1/2, with nothing said at
  all. `Species ~ Sepal.Length + Sepal.Width` on `iris` asked for
  `mtry = 2` of 2, also in silence – every predictor sampled at every
  split, which is bagging rather than a random forest. Neither default
  is computed now: where the caller and the tuner leave the argument
  unset, it is left unset, and the wrapped package applies the same
  default it documents. A `y ~ .` formula is unaffected, which is why
  this survived. The 0.3.0 entry below took the response column out of
  the SVM count; what remained was every other column in the frame.
  Leaving an argument out takes
  [`do.call()`](https://rdrr.io/r/base/do.call.html), which evaluates
  before it builds the call, so the
  [`match.call()`](https://rdrr.io/r/base/match.call.html) these
  backends run recorded the training frame as a literal:
  `print(model$fit)` spilled every row, and on a 960-row frame the
  stored call alone was 159 Kb of a 1.5 Mb forest. The `data` argument
  is put back to a symbol after the fit.

- The `$fit` slot was documented as the wrapped object throughout. That
  holds for a supervised method; an unsupervised one returns tidied
  components as well, so its `$fit` is the list holding them and the
  wrapped object is at `$fit$model`. Corrected in
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md),
  the `Description` field, the README, the architecture notes, and the
  getting-started, unsupervised and integration vignettes.

- [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  raised “subscript out of bounds” for a zero-length or multi-element
  `source`, the one malformed input that missed its own “Invalid S3 URI”
  message.

#### Metrics and evaluation

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  explains a metric that no fold could compute rather than leaving a
  bare `NaN` in the summary. `folds = nrow(data)` is leave-one-out, so
  every test fold holds one observation and `rsq` — which needs
  variation in the truth — is undefined;
  [`mean()`](https://rdrr.io/r/base/mean.html) over nothing then
  reported `NaN`, which reads as a malfunction rather than as a property
  of the request. `rmse` and `mae` are defined for a single observation
  and are unaffected.

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) no
  longer repeats
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)’s
  notes once per fold. The note that a numeric response with few
  distinct values is being treated as regression is about the data, not
  the fold, and appeared k times.

- [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  computed precision, recall, sensitivity, specificity and F1 for the
  **wrong class**. The `yardstick` calls omitted `event_level`, so they
  defaulted to the first factor level while the rest of the package —
  AUC, class prediction, lift and gain — treats the second level as
  positive. A binary model predicting only positives reported
  specificity 1.0 where the true value is 0.0. Threshold metrics from
  [`tl_evaluate_thresholds()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate_thresholds.md)
  were affected the same way, so reported precision fell as the
  threshold rose. Multiclass metrics were never affected.

- [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  never evaluated the last `n %% folds` observations: folds were sized
  with `floor(n / folds)` and sliced forward, leaving the remainder in
  every training set and no test set. On `mtcars` with `folds = 5`, 30
  of 32 rows were scored. Rows are now assigned to folds so that the
  folds partition the data and differ in size by at most one.
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) also
  rejects fold counts below 2 or above `nrow(data)`.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  tested linearity with `cor(fitted, residuals)`, which is identically
  zero for any OLS fit with an intercept — the check could only ever
  report SATISFIED. It is now a RESET-style test on powers of the fitted
  values.

#### Fitting

- `"ridge"`, `"lasso"` and `"elastic_net"` no longer fail when a
  predictor has a missing value. The response was read from `data` while
  the design matrix came from
  [`model.frame()`](https://rdrr.io/r/stats/model.frame.html), which
  applies `na.omit` — so a single missing predictor left `y` one row
  longer than `x`, and glmnet reported “number of observations in y (60)
  not equal to the number of rows of x (59)”. That names neither missing
  values nor the column responsible, and reads as though the caller had
  passed mismatched inputs. The response is now taken from the same
  model frame that builds the design matrix, so these methods drop the
  incomplete row and carry on, as
  [`lm()`](https://rdrr.io/r/stats/lm.html), `rpart()`, `nnet()` and
  `svm()` already did.

#### Prediction

- Classification now reduces the response to the classes it contains. A
  subset keeps every factor level, so `iris[iris$Species != "setosa", ]`
  holds two classes and declares three, and that frame broke seven of
  the eight classification methods in seven different ways:
  `randomForest` and `glmnet` refused to fit, `gbm` and `nnet` failed at
  [`predict()`](https://rdrr.io/r/stats/predict.html) or
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md),
  and `rpart` returned a probability column for the class that was not
  there. Worst of the set,
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  read the declared level count when deciding whether the problem was
  binary, so it stopped passing `event_level` and let `yardstick` score
  the first class as positive — silently reopening, for any such
  response, the metric defect fixed above.

  The fitted models were never wrong:
  [`glm()`](https://rdrr.io/r/stats/glm.html) and the rest drop an empty
  level internally, so the coefficients always matched the explicitly
  dropped frame. Only tidylearn’s description of them was wrong. The
  response is normalised once in
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md),
  so the specification, the fit and every predict path now agree, and
  metrics from a subset match those from
  [`droplevels()`](https://rdrr.io/r/base/droplevels.html) exactly.

- `tl_model(method = "logistic")` records a classification model when
  the response is stored as something other than a factor. A 0/1 integer
  response produced a binomial
  [`glm()`](https://rdrr.io/r/stats/glm.html) described by a
  specification that said `is_classification = FALSE`, so
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  scored it with `rmse`, `mae` and `rsq` — and asking it for `accuracy`
  returned an empty tibble, with no error and no warning.

- [`predict()`](https://rdrr.io/r/stats/predict.html) failed or returned
  wrong output for six method-and-task combinations, all now fixed and
  covered by a contract test that runs every method through the same
  grid:

  - Multiclass `"boost"` returned a **single** prediction for the whole
    input, because `predict.gbm` hands back a 3-D array that
    [`is.matrix()`](https://rdrr.io/r/base/matrix.html) does not
    recognise. `type = "prob"` errored for any input with more than one
    row.
  - `"svm"` with `type = "prob"` always errored: the fitted object
    records the flag as `$compprob`, not `$probability`.
  - Binary classification with `method = "nn"` could not fit at all —
    `entropy` was passed explicitly and collided with the value
    `nnet.formula()` supplies itself.
  - `"xgboost"` built its design matrix from the full two-sided formula,
    so scoring data without the response column was impossible.
  - `"svm"` and `"xgboost"` silently dropped rows with missing
    predictors, returning a shorter vector so that predictions no longer
    lined up with the input rows.
  - Multinomial `"ridge"`/`"lasso"`/`"elastic_net"` with `type = "prob"`
    errored on single-row input.

  The `nn` failure is worth its own note: `nnet.formula()` supplies
  `entropy = TRUE` itself when the response is a two-level factor, and
  [`tl_fit_nn()`](https://tidylearn.sheetsolved.com/reference/tl_fit_nn.md)
  named it again, so `nnet.default()` received it twice and reported
  “formal argument ‘entropy’ matched by multiple actual arguments”.
  Three or more classes were unaffected, because `nnet.formula()` uses
  `softmax` there and `nnet.default()` sets `entropy` to `FALSE`
  whenever `softmax` is on — so the argument it collided with was never
  present. The criterion is now left to nnet. Neural networks had no
  test coverage at all; there are now four tests beyond the contract
  grid.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  model fitted with engineered features no longer errors on raw new
  data. Four of the eight candidates a typical search produces — the
  `pca_*` and `clustered_*` variants — were fitted on columns that exist
  only inside the search, so predicting on a held-out set failed with
  “object ‘PC1’ not found” or “object ‘cluster_kmeans’ not found”.
  Whenever one of those won the leaderboard,
  `predict(result$best_model, new_data = ...)` was unusable. Each
  variant now records the transformation that produced its features, and
  [`predict()`](https://rdrr.io/r/stats/predict.html) replays it —
  fitted on the training data — before dispatching.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a k-means model
  matched `new_data` to the cluster centres by position, taking every
  numeric column in whatever order it arrived. A mismatched width was
  recycled rather than rejected, producing cluster numbers that looked
  valid and were not; a reordered frame silently measured distance
  against the wrong centres. Columns are now matched by name, and a
  missing or non-numeric column is an error naming the column.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on a PCA model had
  the same defect and now aligns `new_data` to the training predictors
  by name.

- `tl_reduce_dimensions(n_components = k)` trimmed its returned data to
  `k` components but left the reduction model projecting onto all of
  them, so `predict(result$reduction_model, new_data)` returned a wider
  matrix than the model trained on `$data` could consume. The component
  budget is now recorded on the model and honoured by
  [`predict()`](https://rdrr.io/r/stats/predict.html).

- XGBoost prediction pins the training factor levels, so new data
  missing a level no longer changes the contrast coding, and no longer
  passes `ntreelimit` or `reshape` to `xgboost::predict()`. Both are
  deprecated upstream and warn that they will become errors; every
  XGBoost prediction emitted two warnings per call.
  [`tl_predict_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_predict_xgboost.md)
  gains `iterationrange` and accepts `ntreelimit` with a deprecation
  warning that translates it. Multiclass probabilities are reshaped to
  one named column per class whichever shape the installed xgboost
  returns.

#### Data leakage

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  learned imputation medians and standardisation centres and scales from
  the **whole** dataset and only then split, so every assessment row
  helped define the transformation it was scored under. Each fold, and
  each side of a train/test split, now learns its own statistics. The
  final model still uses the full-data statistics, which
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  continues to replay.

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  also imputed the **response**, replacing missing outcomes with the
  median and turning them into both training targets and evaluation
  ground truth. Imputation now skips the response.

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  fitted PCA rotations and cluster centroids on all rows before
  cross-validating on the transformed data, so the `pca_*` and
  `clustered_*` candidates competed against honestly scored baselines.
  Both are now refitted inside each fold, via a new `transform` argument
  to [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md).

#### Ranking, splitting and tuning

- [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  also refused a grid naming a single parameter.
  [`expand.grid()`](https://rdrr.io/r/base/expand.grid.html) of one
  parameter is a single-column data frame, and `[i, ]` on one of those
  drops to a bare vector with the column name gone, so the parameters
  reached xgboost unnamed and it stopped with “parameter names cannot be
  empty strings”.
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  had the same slip fixed for 0.4.0; this call site was missed.

- [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  could not complete a run. It read `best_iteration` from the top level
  of the `xgb.cv()` result, which is where xgboost kept it before 3.0
  and not where it has been since, so every parameter set scored `NULL`,
  [`which.min()`](https://rdrr.io/r/base/which.min.html) over those
  scores returned `integer(0)`, and the call died on “attempt to select
  less than one element in get1index” — on the documented default call,
  for any input. Both locations are now read. Separately, `nrounds` was
  hardcoded at 1000 inside the function while `...` was forwarded to the
  same call, so passing the one argument an xgboost tuner obviously
  takes gave “formal argument "nrounds" matched by multiple actual
  arguments”. It is a named argument now, documented as the ceiling
  early stopping works within. The function had no test; it has one now.

- [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  rejects a parameter range written backwards. `c(0.1, 0.001)` instead
  of `c(0.001, 0.1)` was sampled with `runif(1, 0.1, 0.001)`, which is
  `NaN` — and R only warns — so every iteration drew `NaN`, models were
  fitted with `cp = NaN`, and `best_params` was reported as `NaN`
  without anything failing. Equal bounds and a non-positive lower bound
  on a log-uniform range are refused for the same reason.

- [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  accepts a discrete set of numbers that are not whole. Only whole
  numbers reached the discrete branch, so
  `list(cp = c(0.001, 0.01, 0.1))` — the natural way to write candidate
  values for a parameter that is never an integer — was rejected as an
  “Unsupported parameter space definition”, while
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  took the same vector without complaint.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  name a metric they cannot produce. Asking for `"accuracy"` on a
  regression task, or for a metric that does not exist, failed with
  “replacement has length zero” from the assignment that came up empty.
  The error now says which metric was asked for and lists what the task
  does produce.

- `tl_tune_deep(learning_rates = )` searched over a value that changed
  nothing. It passed `optimizer = optimizer_adam(learning_rate = )` into
  [`tl_fit_deep()`](https://tidylearn.sheetsolved.com/reference/tl_fit_deep.md),
  which has no such formal, so the argument fell into `...` and was
  forwarded to
  [`keras::fit()`](https://generics.r-lib.org/reference/fit.html) — by
  which point the model is compiled, and `compile()` is what sets the
  optimizer. Every point on the grid therefore trained at the same rate,
  and `best_learning_rate` was whichever happened to score highest on
  noise.
  [`tl_fit_deep()`](https://tidylearn.sheetsolved.com/reference/tl_fit_deep.md)
  gains a `learning_rate` argument that reaches `compile()`, and the
  final refit on the winning configuration uses it too.

- [`tl_tune_deep()`](https://tidylearn.sheetsolved.com/reference/tl_tune_deep.md)
  reports when no configuration could be fitted. Each fit is wrapped
  individually, so a bad argument forwarded through `...` left every
  `val_loss` as `NA`;
  [`which.min()`](https://rdrr.io/r/base/which.min.html) then returned
  `integer(0)` and the function failed with “attempt to select less than
  one element in get1index”, which describes nothing.

- `tl_auto_ml(metric = "mape")` returned the model with the **highest**
  error as the best one — `mape` was missing from the ascending-sort
  list. Unrecognised metrics now error rather than assume a direction.
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  also returns `best_model_name`.

- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  could return an empty training set *and* an empty test set:
  `floor(n * prop)` can be zero, and `data[-integer(0), ]` selects
  nothing. Every group now keeps at least one row on each side.

- [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  ignored two documented parameter forms. Any two-element numeric was
  caught by the continuous branch first, so an integer range like
  `c(100, 500)` was sampled with
  [`runif()`](https://rdrr.io/r/stats/Uniform.html); and the log-uniform
  form `c(min, max, "log")` is a character vector, so its branch was
  unreachable and the literal `"log"` could be sampled as a value.
  `param_space` is now fully documented.

- [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  accepted a partial `preprocessing` or `evaluation` list and then
  failed inside
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  with “argument is of length zero”. Both specifications now fill in
  their defaults for anything unnamed. An unrecognised name is an error
  rather than a step that silently does nothing, and
  `evaluation$best_metric` is checked against `evaluation$metrics`.

#### Diagnostics

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md),
  [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md)
  and
  [`tl_diagnostic_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_diagnostic_dashboard.md)
  no longer fail when a predictor has a missing value.
  [`lm()`](https://rdrr.io/r/stats/lm.html) drops incomplete cases, so
  [`residuals()`](https://rdrr.io/r/stats/residuals.html),
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html) and every
  influence measure came back shorter than `model$data`, and combining
  them raised “arguments imply differing number of rows: 60, 59” — which
  describes nothing the caller did.

- [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md)
  numbers observations by their row in the training data. It used `1:n`,
  so after a dropped row every observation was attributed to its
  neighbour: with row 3 missing, what the table called observation 3 was
  row 4, and so on to the end.

- `optimal_hclust_k(method = "gap")` never ran.
  [`cluster::clusGap()`](https://rdrr.io/pkg/cluster/man/clusGap.html)
  requires its clustering function to return a list with a `cluster`
  element and [`cutree()`](https://rdrr.io/r/stats/cutree.html) returns
  a bare integer vector, so every call failed with “\$ operator is
  invalid for atomic vectors”. Two further faults sat behind that one
  and could not show themselves while it errored on the first call: the
  refit used [`stats::dist()`](https://rdrr.io/r/stats/dist.html), whose
  default is Euclidean, so a model built with any other distance was
  scored against clusterings it would never produce; and a model built
  from a `dist` object has no observations to resample, which surfaced
  as “no applicable method for ‘select’ applied to an object of class
  NULL” rather than as an explanation. All three are fixed, and the last
  is now an error that says to refit from the data or use
  `"silhouette"`, which works from distances alone.

- [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  converted a `dist` input with
  [`as.matrix()`](https://rdrr.io/r/base/matrix.html) and passed it as
  coordinates, clustering each observation’s vector of distances rather
  than the dissimilarity. It also read a non-existent `"core"`
  attribute, so every point was reported as a non-core point.

- [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md)
  lost its entire metrics tibble for the Lloyd, Forgy and MacQueen
  algorithms, which leave `ifault` NULL.

- [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  documented `weights` as a named vector but indexed it positionally,
  applying weights to the wrong variables. Named weights are now matched
  by name, and a mismatched length errors.

- `tidy_mds(method = "sammon")` and `method = "kruskal"` passed MASS’s
  “zero or negative distance between objects i and j” straight through.
  The cause is duplicated rows, which the message does not say. Both now
  check first and name the offending pairs.

- [`tl_plot_cv_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_cv_results.md)
  could not plot
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  output — it read `$fold_metrics` and `mean_value`, which are named
  `$folds` and `mean`.

- Lift and gain charts indexed past the end of the data in their final
  deciles, corrupting the cumulative curve.

- The outlier plot from
  [`tl_detect_outliers()`](https://tidylearn.sheetsolved.com/reference/tl_detect_outliers.md)
  attached flags to the wrong observations whenever more than one
  variable was plotted.

- [`plot_distance_heatmap()`](https://tidylearn.sheetsolved.com/reference/plot_distance_heatmap.md)
  sorted its axes alphabetically, moving the diagonal off the diagonal
  and discarding any `cluster_order`.

- Influence plots used unnamed colour vectors, so when every point was
  influential they all rendered in the “not influential” colour.

- [`tl_plot_nn_architecture()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_architecture.md)
  failed on any neural network with a single output unit — every
  regression fit, and every two-class fit once those could be fitted at
  all.
  [`NeuralNetTools::plotnet()`](https://rdrr.io/pkg/NeuralNetTools/man/plotnet.html)
  evaluates `mod_in$call$formula` on that branch, and `nnet()` records
  its call verbatim, so what it found was the symbol `formula` resolving
  to [`stats::formula`](https://rdrr.io/r/stats/formula.html): “cannot
  coerce type ‘closure’ to vector of type ‘character’”.
  [`tl_fit_nn()`](https://tidylearn.sheetsolved.com/reference/tl_fit_nn.md)
  now substitutes the formula into the recorded call. Multiclass took
  the other branch, which is why the function’s own example passed.

- `tl_plot_tuning_results(plot_type = "parallel")` and
  [`tl_plot_regularization_path()`](https://tidylearn.sheetsolved.com/reference/tl_plot_regularization_path.md)
  used the `size` aesthetic on a line, which ggplot2 deprecated in 3.4.0
  and which told the user to file a bug against tidylearn. Both use
  `linewidth`.

- `tidy_pca_biplot(color_by = )` and `plot_mds(color_by = )` accepted
  only a column name, but the tibbles they draw from carry an identifier
  and the coordinates — there is nowhere for a grouping variable to
  live, so the documented use was unreachable. Both now also accept a
  vector as long as the data, and a name that cannot resolve is an error
  rather than a plot that fails when printed.

- [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  emitted “essentially perfect fit” warnings from
  [`summary.lm()`](https://rdrr.io/r/stats/summary.lm.html). The slope
  is estimated by regressing the model’s own fitted values on the grid,
  which for a linear model lie exactly on a line, so the warning was
  expected by construction and is no longer passed on. The documentation
  now says that `slopes$slope_se` describes the fit to the prediction
  grid rather than the uncertainty of the marginal effect.

#### Data ingestion

- [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  no longer lets a dataset slug reach the shell as written. The slug was
  interpolated into [`system2()`](https://rdrr.io/r/base/system2.html),
  which applies [`shQuote()`](https://rdrr.io/r/base/shQuote.html) to
  the command and leaves the arguments alone, and
  `tl_parse_kaggle_url()` matched `[^/]+/[^/]+$` — which admits `;`,
  `|`, backticks and `$(`. The URL parser is also skipped entirely when
  the caller passes a bare string, so the slug was not necessarily
  anything Kaggle produced. A pasted dataset link was the vector. Slugs
  and file names are now validated against what Kaggle identifiers
  actually are, before interpolation and before the CLI is looked for,
  and caller-derived values are quoted — which also fixes a destination
  path containing spaces.

- `tl_read_kaggle(file = NULL)` downloaded into a shared
  [`tempdir()`](https://rdrr.io/r/base/tempfile.html) and returned the
  newest matching data file, so a file left by an earlier call could be
  handed back as the requested dataset. Each download now gets its own
  directory, emptied first.

- `tl_read_kaggle(type = "competition")` returned no data. Competition
  downloads arrive zipped and that endpoint has no `--unzip` flag, so
  the search for a data file found none. Archives are unpacked first,
  and the search recurses.

- `tl_read_zip(format = )` forced one format onto every member. A zip
  holding a CSV and a JSON read the JSON as CSV and row-bound the
  result, producing a frame with a column named after the JSON’s first
  line and no error at all. When the archive holds more than one kind of
  data file, `format` now selects the members of that format.

#### Errors instead of misleading results

- Unsupervised routines that cannot use missing values now say so,
  naming the columns and how many values are affected.
  [`tidy_kmeans()`](https://tidylearn.sheetsolved.com/reference/tidy_kmeans.md)
  and
  [`tidy_gap_stat()`](https://tidylearn.sheetsolved.com/reference/tidy_gap_stat.md)
  previously surfaced
  [`stats::kmeans()`](https://rdrr.io/r/stats/kmeans.html)’s “NA/NaN/Inf
  in foreign function call (arg 1)”,
  [`tidy_pca()`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md)
  gave [`prcomp()`](https://rdrr.io/r/stats/prcomp.html)’s “infinite or
  missing values in ‘x’”, and
  [`calc_wss()`](https://tidylearn.sheetsolved.com/reference/calc_wss.md),
  [`optimal_clusters()`](https://tidylearn.sheetsolved.com/reference/optimal_clusters.md)
  and
  [`tidy_silhouette_analysis()`](https://tidylearn.sheetsolved.com/reference/tidy_silhouette_analysis.md)
  loop over k with `purrr`, which wrapped those again into “In index: 2.
  Caused by error in `do_one()`”. None of them named the column, the
  problem, or a way forward. Missing values are the most ordinary thing
  that can be wrong with a data set.

  The message points at `"pam"` and `"clara"`, which accept missing
  values. Those, along with
  [`tidy_dist()`](https://tidylearn.sheetsolved.com/reference/tidy_dist.md),
  [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md),
  [`tidy_mds()`](https://tidylearn.sheetsolved.com/reference/tidy_mds.md)
  and
  [`tidy_hclust()`](https://tidylearn.sheetsolved.com/reference/tidy_hclust.md),
  are unchanged — they handle missing values themselves, and guarding
  them would remove working behaviour rather than improve a message.

- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  no longer rewrite the session’s random stream. Both called
  [`set.seed()`](https://rdrr.io/r/base/Random.html) when given a
  `seed`, so a function seeded for its own reproducibility was also
  deciding what every later
  [`sample()`](https://rdrr.io/r/base/sample.html) or
  [`rnorm()`](https://rdrr.io/r/stats/Normal.html) in the caller’s
  script returned — two scripts differing only in whether they passed
  `seed` diverged everywhere downstream. The stream is restored on exit;
  the seed still does its own job.

- `tidy_pca(method = "princomp")` produced loadings that could not be
  used. [`princomp()`](https://rdrr.io/r/stats/princomp.html) returns a
  `"loadings"` object rather than a plain matrix, and
  [`tibble::as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html)
  read that as a single long vector — 16 values against 4 row names for
  a four-variable PCA — so
  [`get_pca_loadings()`](https://tidylearn.sheetsolved.com/reference/get_pca_loadings.md)
  failed with “Can’t recycle `..1` (size 16) to match `..3` (size 4)”.
  The loadings now match
  [`prcomp()`](https://rdrr.io/r/stats/prcomp.html)’s, up to the sign
  convention.

- The method and the response now have to agree. `"linear"` and
  `"polynomial"` need a numeric response and `"logistic"` needs exactly
  two classes; every other supervised method takes either. A mismatch is
  an error at
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md),
  naming the methods that would fit.

  `tl_model(iris, Species ~ ., method = "linear")` previously succeeded.
  [`lm()`](https://rdrr.io/r/stats/lm.html) estimates from a factor’s
  underlying integer codes — its coefficients are identical to
  regressing on `as.integer(Species)` — so the classes were treated as
  equally spaced points on a scale and
  [`predict()`](https://rdrr.io/r/stats/predict.html) returned numbers
  between them. Nothing failed at any stage, which made this quieter
  than the logistic case: there was no later error to work back from.

  In the other direction,
  `tl_model(mtcars, mpg ~ wt, method = "logistic")` reported that `mpg`
  “has 25 levels”, listed all of them, and recommended classification
  methods for what is plainly a regression problem. It now says the
  response is numeric with 25 distinct values and points at the
  regression methods, while still accepting a two-class response stored
  as 0/1.

- `tl_model(method = "logistic")` now errors when the response does not
  have exactly two levels. `glm(family = binomial)` accepts a
  three-level factor without complaint and fits the first level against
  the other two, so `tl_model(iris, Species ~ ., method = "logistic")`
  returned a model that looked fine and meant nothing. The failure
  surfaced three calls later, at `predict(type = "class")` and
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md),
  both of which reported only that multiclass logistic was “not
  implemented” — by which point the caller had no reason to suspect the
  method choice. The error is now raised at fit time and names the
  methods that do handle more than two classes. A single-level response
  is reported separately.

  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  offered `logistic` as a default candidate for any classification task,
  so without a matching guard a three-level response would now fail the
  whole pipeline rather than one model. It offers `logistic` only for a
  two-level response, as
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  already did.

- [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md),
  [`tl_anomaly_aware()`](https://tidylearn.sheetsolved.com/reference/tl_anomaly_aware.md),
  [`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)
  and
  [`tl_stratified_models()`](https://tidylearn.sheetsolved.com/reference/tl_stratified_models.md)
  default to `supervised_method = "tree"`. The first three defaulted to
  `"logistic"`, which cannot fit a response with more than two levels or
  a numeric one, and the fourth to `"linear"`, which fits
  [`lm()`](https://rdrr.io/r/stats/lm.html) to a factor response and
  returns numbers rather than refusing.
  `tl_anomaly_aware(iris, Species ~ ., response = "Species")` — the
  function’s own documented example — was in the first group. `"tree"`
  handles regression and classification, at any number of classes.

  This changes the model a call produces when `supervised_method` is not
  given. Pass it explicitly to keep the previous behaviour.

- [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  and
  [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md)
  advertised support for `"ridge"`, `"lasso"` and `"elastic_net"`, but
  glmnet provides no residuals, hat values or influence measures. They
  now explain this instead of failing partway through.

- [`plot_cluster_comparison()`](https://tidylearn.sheetsolved.com/reference/plot_cluster_comparison.md)
  and
  [`create_cluster_dashboard()`](https://tidylearn.sheetsolved.com/reference/create_cluster_dashboard.md)
  called `gridExtra` without a
  [`requireNamespace()`](https://rdrr.io/r/base/ns-load.html) guard.

- Database connection strings carried the password into the returned
  object’s `tl_source` attribute — printed on every
  [`print()`](https://rdrr.io/r/base/print.html) and persisted by
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) — into the progress
  message, and into the URL parse error. All are now redacted.

- [`tl_plot_tuning_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tuning_results.md)
  names the valid `plot_type` values in its error instead of reporting
  “Invalid plot_type or insufficient parameters”.

- [`get_pca_variance()`](https://tidylearn.sheetsolved.com/reference/get_pca_variance.md)
  and
  [`get_pca_loadings()`](https://tidylearn.sheetsolved.com/reference/get_pca_loadings.md)
  accept a PCA model from `tl_model(method = "pca")` as well as a
  [`tidy_pca()`](https://tidylearn.sheetsolved.com/reference/tidy_pca.md)
  object. The two representations carry the same tables under different
  names, and the accessors previously took only one of them.

- `inst/examples/unified_workflow.R` reported “Reduced from 4 to 2
  features” after requesting three components, and passed
  `supervised_method = "logistic"` on three-class iris in three places,
  producing convergence warnings. It is now exercised by
  `tests/testthat/test-examples.R`, so it cannot drift again unnoticed.

### Documentation

- Every exported function now carries a runnable example. Thirteen had
  none:
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md),
  [`tl_compare_pipeline_models()`](https://tidylearn.sheetsolved.com/reference/tl_compare_pipeline_models.md),
  [`tl_plot_cv_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_cv_results.md),
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md),
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md),
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md),
  [`tl_plot_nn_tuning()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_tuning.md),
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md),
  [`tl_plot_xgboost_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_tree.md),
  [`tl_plot_xgboost_shap_dependence()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_dependence.md)
  and the three `print` methods. Writing them is what surfaced the
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md)
  defects above.

- [`tl_plot_nn_tuning()`](https://tidylearn.sheetsolved.com/reference/tl_plot_nn_tuning.md)
  documented the wrong input and the wrong plot. It takes the list
  [`tl_tune_nn()`](https://tidylearn.sheetsolved.com/reference/tl_tune_nn.md)
  returns rather than a fitted model — the error message said so, the
  `@param` did not — and it draws a heatmap of the size-by-decay grid,
  not the training history its title claimed.

- `DiagrammeR` is now declared in Suggests.
  [`tl_plot_xgboost_tree()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_tree.md)
  cannot render without it, reaching it through
  [`xgboost::xgb.plot.tree()`](https://rdrr.io/pkg/xgboost/man/xgb.plot.tree.html).

- Corrected five factual errors across the docs: the README claimed ten
  articles where there are eleven; `compute-backends` said eleven
  CPU-only methods and then listed ten, omitting `"polynomial"`;
  `integration-workflows` still documented
  [`tl_semisupervised()`](https://tidylearn.sheetsolved.com/reference/tl_semisupervised.md)
  as defaulting to `supervised_method = "logistic"` after it changed to
  `"tree"`; `tuning-and-pipelines` wrote 6 x 3 = 19 fits; and
  `CONTRIBUTING.md` gave its versioning worked example against 0.3.0,
  telling contributors to open a NEWS heading a release out of date.

- Rewrote `integration-workflows` and `reporting`, which had drifted
  from the register of the other nine articles. Their generic “Best
  Practices” and “Summary” sections are gone, each function now gets a
  sentence on what it buys you and what it costs, and the
  train-then-replay rule that governs all five integration functions is
  stated once up front rather than only in code comments.

- Removed duplicated prose. The overview blurb, the “what tidylearn is /
  is NOT” bullets and the principles list each existed verbatim in two
  or three of README, `getting-started` and `PACKAGE_ARCHITECTURE.md`;
  `PACKAGE_ARCHITECTURE.md` now links to the README for all three, the
  way it already did for the method-to-package table. `getting-started`
  and `supervised-learning` no longer close with a summary restating
  their own introductions.

- New vignette `compute-backends`: how `compute = "auto"` routes a fit,
  what the advisor estimates a cloud tier would cost, and the safety
  model that governs data egress.

- New vignette `market-basket`: the association rules family
  ([`tidy_apriori()`](https://tidylearn.sheetsolved.com/reference/tidy_apriori.md),
  [`inspect_rules()`](https://tidylearn.sheetsolved.com/reference/inspect_rules.md),
  [`filter_rules_by_item()`](https://tidylearn.sheetsolved.com/reference/filter_rules_by_item.md),
  [`find_related_items()`](https://tidylearn.sheetsolved.com/reference/find_related_items.md),
  [`recommend_products()`](https://tidylearn.sheetsolved.com/reference/recommend_products.md),
  [`summarize_rules()`](https://tidylearn.sheetsolved.com/reference/summarize_rules.md),
  [`visualize_rules()`](https://tidylearn.sheetsolved.com/reference/visualize_rules.md))
  had no narrative documentation.

- New vignette `tuning-and-pipelines`:
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md),
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md),
  [`tl_default_param_grid()`](https://tidylearn.sheetsolved.com/reference/tl_default_param_grid.md),
  [`tl_plot_tuning_results()`](https://tidylearn.sheetsolved.com/reference/tl_plot_tuning_results.md)
  and the
  [`tl_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_pipeline.md)
  family, none of which were covered.

- New vignette `diagnostics`:
  [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md),
  [`tl_influence_measures()`](https://tidylearn.sheetsolved.com/reference/tl_influence_measures.md),
  [`tl_detect_outliers()`](https://tidylearn.sheetsolved.com/reference/tl_detect_outliers.md),
  [`tl_diagnostic_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_diagnostic_dashboard.md),
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md),
  [`tl_test_model_difference()`](https://tidylearn.sheetsolved.com/reference/tl_test_model_difference.md),
  [`tl_test_interactions()`](https://tidylearn.sheetsolved.com/reference/tl_test_interactions.md),
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  and
  [`tl_explore()`](https://tidylearn.sheetsolved.com/reference/tl_explore.md).

- `unsupervised-learning` rewritten to use the package’s own `tidy_*()`
  and `augment_*()` interface. It previously reached into
  `model$fit$clusters`, `$fit$centers`, `$fit$loadings` and
  `$fit$variance_explained` throughout, and hand-rolled an elbow search,
  while
  [`optimal_clusters()`](https://tidylearn.sheetsolved.com/reference/optimal_clusters.md),
  [`plot_elbow()`](https://tidylearn.sheetsolved.com/reference/plot_elbow.md),
  [`plot_silhouette()`](https://tidylearn.sheetsolved.com/reference/plot_silhouette.md),
  [`suggest_eps()`](https://tidylearn.sheetsolved.com/reference/suggest_eps.md)
  and
  [`explore_dbscan_params()`](https://tidylearn.sheetsolved.com/reference/explore_dbscan_params.md)
  went unmentioned.

- `automl` now executes. Twenty-three of its twenty-five chunks were
  `eval = FALSE`, with hand-written `#>` lines that read as console
  output and were not. The budget-tier table of predicted model counts
  is replaced by a sweep that measures them.

- `integration-workflows` no longer emits 135 recycling warnings from
  the PCA-then-cluster workflow, and its reported accuracy is no longer
  computed from mis-assigned clusters.

- `supervised-learning` seeds the missing-values example, which was
  unreproducible across builds.

- README links the documentation site and every article; `inst/CITATION`
  reports the installed version and year rather than a hard-coded 2025.

- `inst/security/threat-model.md` is rewritten for the architecture the
  transport spike settled on: plain HTTPS to a Modal Web Function backed
  by an R worker, rather than reticulate driving the Python SDK. T1 and
  T4 named constraints that no longer apply, and no threat covered a
  user-supplied endpoint URL.

### Internal

- Removed four internal helpers with no callers: `create_obs_ids()`,
  `extract_response()`, `get_numeric_cols()` and `validate_data()`. They
  had survived two reviews on the grounds that they looked like
  intentional utilities.

- `.github/workflows/pkgdown.yaml` builds on pull requests without
  deploying, so a dangling article name fails a PR check rather than the
  first push to main, and deploys with `clean: true` so removed pages
  leave the live site.

## tidylearn 0.4.0

CRAN release: 2026-08-03

### New Features

#### Compute backends (foundation)

- [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  — detects local NVIDIA CUDA support and reports which GPU-capable
  backends (xgboost, keras, tensorflow, torch) are installed. Cheap
  detection: parses `nvidia-smi` output and checks installed packages
  without loading Python or fitting a model. Returns a
  `tidylearn_gpu_check` object with a
  [`print()`](https://rdrr.io/r/base/print.html) method.

- [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  — S3 generic that estimates runtime, peak RAM, and cost across local
  CPU, local GPU, and cloud GPU tiers for a given tidylearn method and
  dataset. Dispatches on either a method name (`character`) or a fitted
  `tidylearn_supervised` model. Returns a structured recommendation with
  a [`print()`](https://rdrr.io/r/base/print.html) method. Cloud-tier
  estimates are reported but not yet executable; Modal integration will
  follow in a later iteration.

#### Compute backends (local GPU routing)

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  now accepts a `compute` argument on both supervised and unsupervised
  paths: `"cpu"` (default — existing behaviour), `"gpu"` (route to local
  CUDA when the method supports it), `"auto"` (consult
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  and pick per call), or `"cloud"` (reserved; errors with a clear
  message until the Modal integration lands).

- `tl_fit_xgboost(compute = "gpu")` passes `device = "cuda"` to
  `xgb.train()`. Requires xgboost compiled with CUDA support.

- `tl_fit_deep(compute = "gpu")` defers to TensorFlow’s automatic CUDA
  detection — the argument is accepted for API consistency but does not
  itself change the keras model setup.

- All compute validation flows through `tl_resolve_compute()` so the
  behaviour is uniform across paradigms: methods without an upstream GPU
  path (linear, glm, randomForest, pca, kmeans, etc.) warn and fall back
  to CPU when `"gpu"` is requested; `"cloud"` errors the same way on
  supervised and unsupervised methods. The resolved tier is recorded on
  `model$spec$compute` for both paradigms.

#### Compute backends (cloud reframed as memory-headroom tier)

- [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  now treats cloud as a “doesn’t fit on my machine” tier rather than a
  GPU-acceleration-only tier. Cloud estimates are produced for every
  method the advisor supports (not just GPU-eligible ones), and the
  recommendation flips to `"cloud"` whenever the local job is
  RAM-infeasible — including CPU-only methods like linear regression,
  SVM or random forest on very large data.

  Scope: the advisor covers the 13 supervised methods in
  `.tl_method_profiles`. Unsupervised methods (PCA, k-means, MDS,
  clustering) are not modelled and calling the advisor on one errors.
  Reaching the cloud recommendation through `tl_model(compute = "auto")`
  additionally requires a method with an upstream GPU path (`xgboost`,
  `deep`), since `tl_resolve_compute()` short-circuits CPU-only methods
  to `"cpu"` before consulting the advisor. Call
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  directly to get memory-headroom advice for the other supervised
  methods.

- New internal Modal instance tier table (`.tl_modal_tiers`) listing
  CPU-RAM tiers (`cpu-small`, `cpu-large`, `cpu-xlarge`) alongside GPU
  tiers (`t4`, `a10g`, `a100-40gb`, `a100-80gb`). The advisor picks the
  cheapest viable tier for the workload based on RAM headroom and
  whether the method has an upstream GPU path. Pricing is approximate as
  of early 2026 and may drift; revise if Modal pricing changes.

- The advisor’s recommendation is no longer gated on `cloud$configured`.
  The advisor advises optimally; the caller (`tl_resolve_compute()`)
  decides whether it can act on a cloud recommendation. When
  `compute = "auto"` and the advisor recommends cloud,
  `tl_resolve_compute()` emits a clear message that cloud isn’t yet
  wired up and falls back to local CPU.

- Print method updated: the cloud line now shows the chosen tier label
  (e.g., `T4 (16 GB VRAM / 16 GB RAM)`) alongside the time and cost
  estimate.

#### Compute backends (security threat model)

- Added `inst/security/threat-model.md` — the contract for what cloud
  compute in tidylearn will and will not do once the Modal integration
  lands. Covers token handling (never read in R), data egress consent
  (per-call `confirm_upload = TRUE` plus session-level
  [`tl_cloud_consent()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_consent.md)),
  ephemeral compute (no persistent Modal volumes by default), no
  telemetry, and an audit checklist that reviewers can grep / verify
  against the Modal-integration PR. The doc is shipped with the package
  so users (and CRAN reviewers) can find it via
  `system.file("security/threat-model.md", package = "tidylearn")`.

### Bug Fixes

These four defects produced plausible but wrong numbers rather than
errors, so results computed with earlier versions should be rechecked.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  scored classification models against raw prediction output rather than
  class labels. Because the default
  [`predict()`](https://rdrr.io/r/stats/predict.html) type returns
  probabilities for logistic regression, comparing them to factor labels
  gave an accuracy of exactly 0 for every logistic model. Evaluation now
  requests `type = "class"` explicitly. Everything built on
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  was affected —
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md),
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md),
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md),
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md),
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  and
  [`tl_compare_cv()`](https://tidylearn.sheetsolved.com/reference/tl_compare_cv.md)
  all ranked logistic models last regardless of how they actually
  performed.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  had no `metrics` argument, so a requested metric silently landed in
  `...` and was forwarded to
  [`predict()`](https://rdrr.io/r/stats/predict.html). Only accuracy
  (classification) or rmse/mae/rsq (regression) were ever returned.
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  now takes `metrics` and computes the requested set, delegating to
  [`tl_calc_classification_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_calc_classification_metrics.md)
  for classification. Classification supports accuracy, precision,
  recall, sensitivity, specificity, f1, auc and pr_auc; regression
  supports rmse, mse, mae, mape and rsq.
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md)
  gains a matching `metrics` argument. This removes the “Could not
  determine best model … all values NA” warning from default pipeline
  runs and the `replacement has length zero` error from
  `tl_tune_grid(metric = "f1")`.

  Regression `rsq` is now `1 - SS_res/SS_tot` rather than the squared
  correlation. The two agree for in-sample OLS; the squared correlation
  was optimistic on held-out data.

- [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  derived its centre and scale from `results$processed_data`, which is
  stored *after* standardization — so new data was rescaled against a
  mean of ~0 and an sd of ~1 and reached the model in raw units. On
  `mtcars` with `mpg ~ wt + hp` this returned predictions near -230 for
  rows whose actual mpg was 21. The same defect made imputation
  substitute a standardized median (~0) for missing values instead of
  the raw-scale one.
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  now records the medians, modes, centres and scales it learned in
  `results$preprocessing_stats`, and
  [`tl_predict_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)
  applies those. Pipelines run by an earlier version carry no such
  statistics and now raise a clear error asking for a re-run rather than
  silently producing wrong predictions. Constant columns are centred
  without dividing by zero.

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)’s
  leaderboard scores were always `NA`. `create_leaderboard()` expected a
  result shape that neither
  [`tl_cv()`](https://tidylearn.sheetsolved.com/reference/tl_cv.md) nor
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  produces, so every model scored `NA` and the reported “best model” was
  whichever trained first. Score extraction now handles both shapes, and
  the target metric is passed through to every evaluation.

- [`predict()`](https://rdrr.io/r/stats/predict.html) on unsupervised
  models used `nrow(new_data) == nrow(object$data)` to decide whether
  new data had been supplied. Any new data with the same number of rows
  as the training set silently got the training result back — verified
  with a PCA projection of an all-999 frame returning the training
  scores. [`predict()`](https://rdrr.io/r/stats/predict.html) now tracks
  whether the caller supplied `new_data` rather than inferring it from
  row count. This also affected
  [`predict.tidylearn_transfer()`](https://tidylearn.sheetsolved.com/reference/predict.tidylearn_transfer.md)
  and
  [`predict.tidylearn_stratified()`](https://tidylearn.sheetsolved.com/reference/predict.tidylearn_stratified.md),
  which delegate to it.

  Methods with no out-of-sample projection (PAM, CLARA, MDS, DBSCAN,
  hierarchical clustering) now error when handed new data instead of
  returning training assignments that look like predictions. PAM and
  CLARA gained the training-data branch they previously lacked, and
  hierarchical clustering — whose fit holds a tree, not assignments —
  points at
  [`tidy_cutree()`](https://tidylearn.sheetsolved.com/reference/tidy_cutree.md)
  rather than returning `NULL`.

- Prediction for `ridge`, `lasso` and `elastic_net` built its design
  matrix from a `~ predictors - 1` formula while the fit used
  [`model.matrix()`](https://rdrr.io/r/stats/model.matrix.html) with the
  intercept dropped. The two disagree whenever a factor predictor is
  present: the fit uses treatment contrasts (k-1 columns), prediction
  one-hot encodes (k columns), so any such model failed with
  `The number of variables in newx must be N`. The fit now records its
  terms and factor levels, and prediction rebuilds an identically-coded
  design matrix from them.

- Regularized classification ignored the `type` argument and always
  returned class labels, so `type = "prob"` gave labels and ROC,
  calibration, lift and gain plots could not work for these models.
  `type = "prob"` now returns one probability column per class (binary
  and multinomial), and `type = "class"`/`"response"` returns a factor
  carrying the training levels rather than a character vector. An
  unrecognised type errors instead of silently returning labels.

- `method = "boost"` could not fit a classification model at all:
  `gbm()` was handed a factor response with
  `distribution = "bernoulli"`, which requires a numeric 0/1 response.
  The response is now encoded with the second factor level as the
  positive class, matching the orientation
  [`tl_predict_boost()`](https://tidylearn.sheetsolved.com/reference/tl_predict_boost.md)
  already assumed.

- [`plot()`](https://rdrr.io/r/graphics/plot.default.html) failed for
  every unsupervised method. The `tl_fit_*` wrappers unpack the `tidy_*`
  objects into plain lists, but the plot helpers were handed the
  unpacked list: k-means, PAM, CLARA and DBSCAN partial-matched
  `$cluster` to the `$clusters` tibble and built a nested column; PCA
  and MDS hit `tidy_pca`/`tidy_mds` class checks that a plain list
  cannot satisfy; hclust passed a list where an `hclust` object was
  expected. Each method now supplies the structure its plot helper
  expects.

#### Compute backends (corrections)

- `parallel` is now declared in Imports.
  `tl_estimate_local_cpu_internal()` calls
  [`parallel::detectCores()`](https://rdrr.io/r/parallel/detectCores.html),
  which without the declaration produces an “‘::’ call not declared
  from” NOTE under `R CMD check`.

- `testthat` minimum raised to 3.1.7. The compute tests use
  `local_mocked_bindings()` (3.1.7) and `expect_no_warning()` (3.1.5);
  on an older testthat the suite errored rather than skipped.

- `tl_detect_cuda_internal()` now checks the exit status of
  `nvidia-smi`. A machine with the binary installed but the driver
  unloaded prints its error message to stdout and exits non-zero — that
  text was being parsed as a device name, so
  [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  reported a working GPU and `compute = "gpu"` routed `device = "cuda"`
  into a fit that then failed.

- GPU routing for xgboost now requires xgboost \>= 2.0.0, checked during
  backend detection. The `device` parameter arrived in 2.0.0; older
  versions ignore unknown parameters, so the fit ran on CPU while
  `spec$compute` recorded `"gpu"`. Older versions are now reported as
  having no GPU path, so `compute = "gpu"` warns and falls back
  honestly.

- `tl_model(compute = "auto")` now forwards the caller’s
  runtime-relevant hyperparameters to the advisor. Previously the
  advisor always estimated a default-sized job, so
  `tl_model(..., method = "xgboost", nrounds = 5000, compute = "auto")`
  was costed as `nrounds = 100` and could choose CPU when GPU was the
  right call.

- [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  no longer skips a local GPU that finishes quickly. The guard required
  an estimated GPU runtime of at least 5 seconds on top of a 3x speedup,
  so a job estimated at 70s on CPU and 4.7s on GPU — a 15x speedup — was
  reported as “No meaningfully faster tier available”. The sub-60s check
  earlier in the same function already covers jobs too small to bother
  offloading.

- `tl_compute_advisor(fitted_model, formula = ...)` no longer errors
  with “formal argument ‘formula’ matched by multiple actual arguments”.
  The documentation says `formula` is ignored for a fitted model; now it
  actually is.

### Other Changes

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  now cross-validates the PCA-augmented and cluster-augmented variants
  when the budget allows. Previously these were scored on training data
  while baselines were cross-validated, so once scoring worked at all,
  overfit variants would have outranked honestly-scored models. The
  leaderboard gains an `evaluation` column recording `"cv"` or `"train"`
  per model, since mixed scores are not directly comparable.

- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  no longer fits logistic regression to a multiclass response — the
  implementation is binary-only, and the resulting model was
  meaningless. It errors early when the response has fewer than two
  observed classes.

- [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  rejects an unnamed `models` argument. Passing a character vector
  previously trained nothing and failed later with an indexing error.

- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  errors when the response column is absent from `new_data` instead of
  computing metrics against `NULL`.

- [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md)
  failed with “argument is of length zero” whenever a `metric` was named
  without also naming `maximize`. The optimisation direction was only
  assigned inside the branch that supplies a default metric, so an
  explicit metric left `maximize` at `NULL` and the later
  `if (maximize)` errored. Direction now follows the metric itself:
  `rmse`, `mse`, `mae` and `mape` are minimised, everything else
  maximised. An explicitly supplied `maximize` is still respected.

- Tuning a single hyperparameter dropped its name. Indexing one column
  of the results without `drop = FALSE` collapsed the row to a bare
  value, so the winning setting was passed to
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
  positionally and never reached the underlying fit — a tuned `cp` or
  `lambda` was silently discarded. Affected both
  [`tl_tune_grid()`](https://tidylearn.sheetsolved.com/reference/tl_tune_grid.md)
  and
  [`tl_tune_random()`](https://tidylearn.sheetsolved.com/reference/tl_tune_random.md).

- `tl_plot_tuning_results(plot_type = "importance")` errored on
  categorical parameters with “Can’t subset `.data` outside of a data
  mask context”. The ANOVA branch built its formula with the tidy-eval
  `.data` pronoun, which [`aov()`](https://rdrr.io/r/stats/aov.html)
  cannot evaluate; it now uses
  [`stats::reformulate()`](https://rdrr.io/r/stats/delete.response.html).

- `tl_plot_tuning_results(plot_type = "grid")` errored with “object ‘p’
  not found” when a parameter had more than 20 unique values. The
  fallback to a scatter plot called the function recursively but
  discarded the result.

### Tests

- New `test-metrics.R` and `test-pipeline.R` cover the four fixes above;
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
  and the whole pipeline family previously had no test coverage, which
  is why the defects survived. Added leaderboard scoring and ranking
  tests to `test-workflows.R`.

- `tl_auto_ml handles small datasets` used `iris[1:30, ]`, which is
  entirely setosa. It passed only because a degenerate single-class
  logistic model was counted as a trained model. It now samples across
  all three species, and a separate test covers the single-class
  rejection.

- New `test-supervised-predict.R` and `test-unsupervised-predict.R`
  cover the prediction fixes above, and `tests/testthat/setup.R` draws
  base-graphics test plots to a null device so they no longer leave an
  `Rplots.pdf` behind.

### Documentation

- Corrected vignette examples that printed wrong results. The
  integration-workflows vignette reported 0% accuracy in five places —
  it compared logistic regression’s probability output against factor
  labels, on a three-class response that logistic regression cannot
  represent. The supervised-learning vignette reported 33.3% (chance)
  for its complete-workflow example, which fitted on standardized
  features and then predicted on raw test data. Both now use
  multiclass-capable methods, score through
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md),
  and apply the training preprocessing to the test set.

- The getting-started and supervised-learning vignettes now explain that
  [`predict()`](https://rdrr.io/r/stats/predict.html)’s default
  `type = "response"` returns probabilities for logistic regression but
  class labels for trees and forests, and show `type = "class"` and
  `type = "prob"` alongside
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md).

- Re-enabled seven vignette chunks that were disabled while the
  underlying bugs were present: ridge, lasso, elastic net and SVM in the
  supervised-learning vignette, and PAM, DBSCAN and CLARA in the
  unsupervised-learning vignette.

- Added package-level documentation, so
  [`?tidylearn`](https://tidylearn.sheetsolved.com/reference/tidylearn-package.md)
  now resolves.

- README: fixed a [`predict()`](https://rdrr.io/r/stats/predict.html)
  example that referenced columns which do not exist, replaced a
  [`plot_clusters()`](https://tidylearn.sheetsolved.com/reference/plot_clusters.md)
  call that passed a model where a data frame is required, and added a
  section on the compute backends.

- [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  documents the `$preprocessing_stats` component, and
  [`predict()`](https://rdrr.io/r/stats/predict.html) no longer
  advertises unsupervised `type` values that it ignores — its `@return`
  now describes the shape unsupervised models actually produce, and
  which of them accept `new_data`.

- [`tl_check_gpu()`](https://tidylearn.sheetsolved.com/reference/tl_check_gpu.md)
  and
  [`tl_compute_advisor()`](https://tidylearn.sheetsolved.com/reference/tl_compute_advisor.md)
  examples now run rather than sitting in `\dontrun{}`; neither requires
  a GPU.

## tidylearn 0.3.1

CRAN release: 2026-05-19

### Performance

- [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  — eliminated two layers of redundant work in the pairwise distance
  loop:
  - Column ranges (`max - min`) and ordinal rank vectors were previously
    recomputed on every `(i, j)` pair. They are now computed once in a
    pre-pass, reducing work from O(n² × p) to O(n² + p).
  - Replaced scalar data-frame indexing `data[i, k]` — which dispatches
    to the R-level `[.data.frame` method on every call — with
    pre-extracted plain-vector access `col_vecs[[k]][i]`, which resolves
    at the C level. Benchmarks show 10–100× faster scalar access; the
    gain compounds across the full `n*(n-1)/2 * p` iterations.
  - Column types (`is.numeric`, `is.ordered`) are now resolved once into
    a `col_type` character vector, removing repeated S3 predicate calls
    from the inner loop.

### Bug Fixes

- Fixed
  [`tl_reduce_dimensions()`](https://tidylearn.sheetsolved.com/reference/tl_reduce_dimensions.md)
  returning the internal `.obs_id` row identifier as a column of its
  `$data` result. Passing that data to a supervised model via a
  `response ~ .` formula fed `.obs_id` in as a high-cardinality
  predictor, which made tree-based fits effectively non-terminating. The
  identifier is now dropped from the returned data, consistent with how
  the pipeline and transfer-learning paths already handle it.
- Fixed [`print()`](https://rdrr.io/r/base/print.html) and
  [`summary()`](https://rdrr.io/r/base/summary.html) erroring on the
  model objects returned by
  [`tl_step_selection()`](https://tidylearn.sheetsolved.com/reference/tl_step_selection.md)
  and
  [`tl_tune_xgboost()`](https://tidylearn.sheetsolved.com/reference/tl_tune_xgboost.md).
  Both constructed their object without the `spec$paradigm` field or the
  `tidylearn_supervised` class, so the print method hit a zero-length
  `if` condition and [`summary()`](https://rdrr.io/r/base/summary.html)
  took the unsupervised branch. Both objects are now built consistently
  with
  [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md).
- Fixed
  [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  (and `tidy_dist(..., method = "gower")`) erroring on single-row input.
  The pairwise loop used `1:(n - 1)`, which produces the invalid
  sequence `1:0` when `n` is 1; it now uses `seq_len(n - 1)`, so a
  single-row data frame returns an empty `dist` object, consistent with
  [`stats::dist()`](https://rdrr.io/r/stats/dist.html).

### Tests

- Added 11 tests for
  [`tidy_gower()`](https://tidylearn.sheetsolved.com/reference/tidy_gower.md)
  / `tidy_dist(..., method = "gower")` covering: return type and
  metadata, symmetry and self-distance, identical rows, hand-verified
  numeric / categorical / ordered / mixed-type distances, NA skipping,
  custom weights, constant-column denominator behaviour, and single-row
  input.

### Internal

- Removed seven unused packages from `Suggests` (caret, mclust, onnx,
  parsnip, recipes, reticulate, workflows) — none were referenced in
  package code, tests, or vignettes.

## tidylearn 0.3.0

CRAN release: 2026-04-09

### New Features

#### Data Ingestion (`tl_read()` Family)

- New
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  dispatcher function — auto-detects format from file extension, URL
  pattern, or connection string and routes to the appropriate reader
- All readers return a `tidylearn_data` object, a tibble subclass
  carrying source, format, and timestamp metadata via
  [`print.tidylearn_data()`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_data.md)

##### File Format Readers

- [`tl_read_csv()`](https://tidylearn.sheetsolved.com/reference/tl_read_csv.md)
  /
  [`tl_read_tsv()`](https://tidylearn.sheetsolved.com/reference/tl_read_tsv.md)
  — via readr with base R fallback
- [`tl_read_excel()`](https://tidylearn.sheetsolved.com/reference/tl_read_excel.md)
  — `.xls`, `.xlsx`, `.xlsm` files via readxl
- [`tl_read_parquet()`](https://tidylearn.sheetsolved.com/reference/tl_read_parquet.md)
  — via nanoparquet
- [`tl_read_json()`](https://tidylearn.sheetsolved.com/reference/tl_read_json.md)
  — tabular JSON via jsonlite
- [`tl_read_rds()`](https://tidylearn.sheetsolved.com/reference/tl_read_rds.md)
  /
  [`tl_read_rdata()`](https://tidylearn.sheetsolved.com/reference/tl_read_rdata.md)
  — native R formats via base R

##### Database Readers

- [`tl_read_db()`](https://tidylearn.sheetsolved.com/reference/tl_read_db.md)
  — query any live DBI connection
- [`tl_read_sqlite()`](https://tidylearn.sheetsolved.com/reference/tl_read_sqlite.md)
  — auto-connect to SQLite files via RSQLite
- [`tl_read_postgres()`](https://tidylearn.sheetsolved.com/reference/tl_read_postgres.md)
  — connection string or named params via RPostgres
- [`tl_read_mysql()`](https://tidylearn.sheetsolved.com/reference/tl_read_mysql.md)
  — connection string or named params via RMariaDB
- [`tl_read_bigquery()`](https://tidylearn.sheetsolved.com/reference/tl_read_bigquery.md)
  — Google BigQuery via bigrquery

##### Cloud/API Readers

- [`tl_read_s3()`](https://tidylearn.sheetsolved.com/reference/tl_read_s3.md)
  — download and read from S3 URIs via paws.storage
- [`tl_read_github()`](https://tidylearn.sheetsolved.com/reference/tl_read_github.md)
  — download raw files from GitHub repositories
- [`tl_read_kaggle()`](https://tidylearn.sheetsolved.com/reference/tl_read_kaggle.md)
  — download datasets via the Kaggle CLI

##### Multi-File Reading

- [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  accepts a character vector of paths — reads each and row-binds with a
  `source_file` column
- [`tl_read_dir()`](https://tidylearn.sheetsolved.com/reference/tl_read_dir.md)
  — scan a directory for data files with optional format, pattern, and
  recursive filtering
- [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md)
  — extract and read from zip archives, with optional file selection
- All backend packages are suggested dependencies, checked at call time
  via `tl_check_packages()`

#### New Vignette

- Added “Data Ingestion with tidylearn” vignette covering all readers,
  databases, cloud sources, multi-file reading, and the full pipeline
- Updated “Getting Started” vignette to include
  [`tl_read()`](https://tidylearn.sheetsolved.com/reference/tl_read.md)
  in the workflow

### Bug Fixes

#### Workflow and Pipeline Fixes

- Fixed
  [`tl_transfer_learning()`](https://tidylearn.sheetsolved.com/reference/tl_transfer_learning.md)
  hanging indefinitely when used with PCA pre-training. The `.obs_id`
  row-identifier column from PCA output was being included in the
  supervised formula, creating a massive dummy-variable matrix. The
  column is now stripped before both training and prediction.
- Fixed
  [`tl_run_pipeline()`](https://tidylearn.sheetsolved.com/reference/tl_run_pipeline.md)
  failing with “attempt to select less than one element” when all
  cross-validation metrics were NA. Root cause:
  [`scale()`](https://rdrr.io/r/base/scale.html) returned matrix columns
  instead of vectors, causing downstream metric computation to produce
  NaN. Added [`as.vector()`](https://rdrr.io/r/base/vector.html) wrapper
  and hardened the best-model selection to handle all-NA metric values
  gracefully.
- Overhauled
  [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  time budget enforcement. The budget now controls which models are
  attempted: budgets under 30s skip slow C-level models (forest, SVM,
  XGBoost) entirely, and cross-validation is skipped when remaining time
  is tight. Baseline model order changed to fast-first (tree,
  logistic/linear, then forest). See
  [`?tl_auto_ml`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md)
  for full details on budget tiers.

#### Interaction and Prediction Fixes

- Fixed
  [`tl_interaction_effects()`](https://tidylearn.sheetsolved.com/reference/tl_interaction_effects.md)
  crashing with “unused argument (se.fit)” because tidylearn’s
  [`predict()`](https://rdrr.io/r/stats/predict.html) method does not
  support `se.fit`. Now uses
  [`stats::predict()`](https://rdrr.io/r/stats/predict.html) on the raw
  model object for confidence intervals. Also fixed an invalid formula
  in the internal slope calculation.
- Fixed
  [`tl_plot_interaction()`](https://tidylearn.sheetsolved.com/reference/tl_plot_interaction.md)
  expecting `fit`/`lwr`/`upr` columns from
  [`predict()`](https://rdrr.io/r/stats/predict.html) output. Now
  correctly handles tidylearn’s `.pred` tibble format.

#### Visualization Fixes

- Fixed
  [`tl_plot_intervals()`](https://tidylearn.sheetsolved.com/reference/tl_plot_intervals.md)
  calling non-existent `tl_prediction_intervals()` function. Now
  computes confidence and prediction intervals directly via
  `stats::predict(..., interval = "confidence")` and
  `stats::predict(..., interval = "prediction")`.
- Fixed
  [`tl_plot_svm_boundary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_svm_boundary.md)
  erroring with “at least two predictor variables required” when using
  `response ~ .` formulas. The function now resolves predictors from
  data column names instead of
  [`all.vars()`](https://rdrr.io/r/base/allnames.html), which does not
  expand `.`. Also switched from `geom_contour_filled` (which failed on
  discrete class predictions) to `geom_raster`.
- Fixed
  [`tl_plot_svm_tuning()`](https://tidylearn.sheetsolved.com/reference/tl_plot_svm_tuning.md)
  passing `NULL` entries in the `ranges` list to
  [`e1071::tune()`](https://rdrr.io/pkg/e1071/man/tune.html), which
  caused “NA/NaN/Inf in foreign function call” errors. Tuning ranges are
  now built conditionally based on the kernel type.
- Fixed
  [`tl_plot_xgboost_shap_summary()`](https://tidylearn.sheetsolved.com/reference/tl_plot_xgboost_shap_summary.md)
  failing with “arguments imply differing number of rows” when
  `n_samples` differed from `nrow(data)`. Sampling is now performed
  before SHAP computation so that feature values and SHAP values always
  have the same number of rows.

#### Other Fixes

- Fixed classification auto-detection silently treating numeric
  responses with \<= 10 unique values as classification. The response
  must now be a factor or character for classification; a helpful
  message is emitted when a low-cardinality numeric response is
  detected.
- Fixed
  [`tl_check_assumptions()`](https://tidylearn.sheetsolved.com/reference/tl_check_assumptions.md)
  crashing with “list object cannot be coerced to logical” when some
  assumption checks returned NULL (e.g., when optional test packages
  were not installed).
- Fixed SVM default `gamma` calculation to use predictor count only
  (`1 / (ncol(data) - 1)`) instead of including the response column.
- Added missing `@return` tag to
  [`print.tidylearn_data()`](https://tidylearn.sheetsolved.com/reference/print.tidylearn_data.md).
- Replaced deprecated ggplot2 `size` parameter with `linewidth` in all
  [`geom_line()`](https://ggplot2.tidyverse.org/reference/geom_path.html)
  calls across visualization, classification, PCA, DBSCAN, and
  validation plotting functions.

### Tests

- Added test suite for visualization module (26 tests) — plot dispatch,
  regression/classification plots, lift/gain charts, model comparison,
  unsupervised visualization, and Shiny dashboard.
- Added test suite for tuning module (49 tests) —
  `tl_default_param_grid`, `tl_tune_grid`, `tl_tune_random`,
  `tl_plot_tuning_results`, and input validation.
- Added test suite for diagnostics module (75 tests) — influence
  measures, influence plots, assumption checking, and outlier detection
  across all methods (IQR, z-score, Cook’s, Mahalanobis).

### Code Quality

- Package-wide lint cleanup — all R source files, tests, and vignettes
  now pass lintr with zero issues
- Replaced unsafe `1:n` patterns with
  [`seq_len()`](https://rdrr.io/r/base/seq.html) /
  [`seq_along()`](https://rdrr.io/r/base/seq.html)
- Removed unused variables across the codebase
- Renamed non-snake_case variables to follow R conventions
- Added `.lintr` configuration enforcing `%>%` pipe consistency

## tidylearn 0.2.0

CRAN release: 2026-03-16

### New Features

#### Formatted gt Tables

- New
  [`tl_table()`](https://tidylearn.sheetsolved.com/reference/tl_table.md)
  dispatcher function — mirrors
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) but produces
  formatted `gt` tables instead of ggplot2 visualisations
- [`tl_table_metrics()`](https://tidylearn.sheetsolved.com/reference/tl_table_metrics.md)
  — styled evaluation metrics table from
  [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
- [`tl_table_coefficients()`](https://tidylearn.sheetsolved.com/reference/tl_table_coefficients.md)
  — model coefficients with p-values (lm/glm) or sorted by magnitude
  (glmnet), with conditional highlighting
- [`tl_table_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_table_confusion.md)
  — confusion matrix with correct predictions highlighted on the
  diagonal
- [`tl_table_importance()`](https://tidylearn.sheetsolved.com/reference/tl_table_importance.md)
  — ranked feature importance with colour gradient
- [`tl_table_variance()`](https://tidylearn.sheetsolved.com/reference/tl_table_variance.md)
  — PCA variance explained with cumulative % coloured
- [`tl_table_loadings()`](https://tidylearn.sheetsolved.com/reference/tl_table_loadings.md)
  — PCA loadings with diverging red–blue colour scale
- [`tl_table_clusters()`](https://tidylearn.sheetsolved.com/reference/tl_table_clusters.md)
  — cluster sizes and mean feature values for kmeans, pam, clara,
  dbscan, and hclust models
- [`tl_table_comparison()`](https://tidylearn.sheetsolved.com/reference/tl_table_comparison.md)
  — side-by-side multi-model comparison table
- All table functions share a consistent `gt` theme via internal
  `tl_gt_theme()` helper
- `gt` is a suggested dependency — functions error with an install
  message if `gt` is not available

#### New Vignette

- Added “Reporting with tidylearn” vignette covering all plot and table
  functions

### Bug Fixes

- Fixed `tl_fit_dbscan()` returning a non-existent `core_points` field
  instead of `summary` from the underlying
  [`tidy_dbscan()`](https://tidylearn.sheetsolved.com/reference/tidy_dbscan.md)
  result

## tidylearn 0.1.1

CRAN release: 2026-03-13

### Bug Fixes

- Fixed [`plot()`](https://rdrr.io/r/graphics/plot.default.html) failing
  on supervised models with “could not find function ‘tl_plot_model’” by
  implementing the missing
  [`tl_plot_model()`](https://tidylearn.sheetsolved.com/reference/tl_plot_model.md)
  and
  [`tl_plot_unsupervised()`](https://tidylearn.sheetsolved.com/reference/tl_plot_unsupervised.md)
  internal dispatchers
  ([\#1](https://github.com/ces0491/tidylearn/issues/1))
- Fixed
  [`tl_plot_actual_predicted()`](https://tidylearn.sheetsolved.com/reference/tl_plot_actual_predicted.md),
  [`tl_plot_residuals()`](https://tidylearn.sheetsolved.com/reference/tl_plot_residuals.md),
  and
  [`tl_plot_confusion()`](https://tidylearn.sheetsolved.com/reference/tl_plot_confusion.md)
  failing due to accessing a non-existent `$prediction` column on
  predict output (correct column is `$.pred`)
- Fixed the same `$prediction` column mismatch in the
  [`tl_dashboard()`](https://tidylearn.sheetsolved.com/reference/tl_dashboard.md)
  predictions table

## tidylearn 0.1.0

CRAN release: 2026-02-06

### Initial CRAN Release

- First release of tidylearn - a unified tidy interface to R’s machine
  learning ecosystem

#### Features

##### Unified Interface

- [`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md) -
  Single function to fit 20+ machine learning models
- Consistent function signatures across all methods
- Tidy tibble output for all results
- Access raw model objects via `$fit` for package-specific functionality

##### Supervised Learning Methods

- Linear regression (stats::lm)
- Polynomial regression (stats::lm with poly)
- Logistic regression (stats::glm)
- Ridge, LASSO, elastic net (glmnet)
- Decision trees (rpart)
- Random forests (randomForest)
- Gradient boosting (gbm)
- XGBoost (xgboost)
- Support vector machines (e1071)
- Neural networks (nnet)
- Deep learning (keras, optional)

##### Unsupervised Learning Methods

- Principal Component Analysis (stats::prcomp)
- Multidimensional Scaling (stats, MASS, smacof)
- K-means clustering (stats::kmeans)
- PAM clustering (cluster::pam)
- CLARA clustering (cluster::clara)
- Hierarchical clustering (stats::hclust)
- DBSCAN (dbscan)

##### Additional Features

- [`tl_split()`](https://tidylearn.sheetsolved.com/reference/tl_split.md) -
  Train/test splitting with stratification support
- [`tl_prepare_data()`](https://tidylearn.sheetsolved.com/reference/tl_prepare_data.md) -
  Data preprocessing (scaling, imputation, encoding)
- [`tl_evaluate()`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md) -
  Model evaluation with multiple metrics
- [`tl_auto_ml()`](https://tidylearn.sheetsolved.com/reference/tl_auto_ml.md) -
  Automated machine learning
- `tl_tune()` - Hyperparameter tuning with grid and random search
- Unified ggplot2-based visualization functions
- Integration workflows combining supervised and unsupervised learning

#### Wrapped Packages

tidylearn wraps established R packages including: stats, glmnet,
randomForest, xgboost, gbm, e1071, nnet, rpart, cluster, dbscan, MASS,
and smacof.
