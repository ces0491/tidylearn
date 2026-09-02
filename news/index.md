# Changelog

## tidylearn 0.5.0

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
  the README, the architecture notes, and the getting-started,
  unsupervised and integration vignettes.

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
