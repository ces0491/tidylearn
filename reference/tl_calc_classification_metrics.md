# Calculate classification metrics

Scores predicted classes, and optionally class probabilities, against
observed classes.

## Usage

``` r
tl_calc_classification_metrics(
  actuals,
  predicted,
  predicted_probs = NULL,
  metrics = c("accuracy", "precision", "recall", "f1", "auc"),
  thresholds = NULL,
  ...
)
```

## Arguments

- actuals:

  Observed classes: a factor, or a character, logical or numeric vector.

- predicted:

  Predicted classes, one for each element of `actuals`.

- predicted_probs:

  Class probabilities, needed for `"auc"`, `"pr_auc"` and `thresholds`:
  a data frame (or matrix) with a row for each element of `actuals` and
  a column for each class, named by class – the shape
  `predict(model, type = "prob")` returns. Without it, `"auc"` and
  `"pr_auc"` are left out of the result, with a warning when they were
  asked for by name.

- metrics:

  Character vector of metrics to compute, from `"accuracy"`,
  `"precision"`, `"recall"`, `"sensitivity"`, `"specificity"`, `"f1"`,
  `"auc"` and `"pr_auc"`. For more than two classes, `"precision"`,
  `"recall"`, `"specificity"` and `"f1"` are macro averages, and `"auc"`
  and `"pr_auc"` average the one-vs-rest areas of the classes.

- thresholds:

  Optional numeric vector of cut-offs on the positive class's
  probability, for binary classification. Each adds rows scoring the
  classes that cut-off assigns. Needs `predicted_probs`.

- ...:

  Not used.

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with
columns `metric` (character) and `value` (numeric), one row per
requested metric. For more than two classes, `"auc"` is followed by an
`auc_<class>` row for each class.

`"auc"` and `"pr_auc"` need at least two classes among the scored rows,
and are `NA`, with a warning, when there is only one. A class with no
scored row has no one-vs-rest area: its `auc_<class>` row is `NA`, the
averages cover the other classes, and a warning names it.

With `thresholds`, six rows per cut-off follow – for a cut-off of 0.5,
`accuracy_t0.5`, `precision_t0.5`, `recall_t0.5`, `f1_t0.5`, `f2_t0.5`
and `f0.5_t0.5` – and the tibble gains a `threshold` column, `NA` on the
other rows.

## Details

The classes, in order, are the levels of `predicted` when it is a factor
of two or more levels – which is how
[`predict()`](https://rdrr.io/r/stats/predict.html) returns them, in the
model's order – and otherwise the classes present in `actuals`. Any
other class found in `actuals` or `predicted` follows them. For two
classes the second is the positive class.
[`tl_evaluate`](https://tidylearn.sheetsolved.com/reference/tl_evaluate.md)
instead leaves out rows of a class the model was never trained on, since
it knows the model's classes.

A row missing its observed class, its prediction or one of its
probabilities is dropped before anything is computed, so every metric
describes the same rows. With no row left – `actuals` empty, or every
row incomplete – there is nothing to score, and it is an error of class
`tidylearn_no_scored_rows`.

## Examples

``` r
# \donttest{
model <- tl_model(iris, Species ~ ., method = "forest")
preds <- predict(model)
tl_calc_classification_metrics(iris$Species, preds$.pred)
#> # A tibble: 4 × 2
#>   metric    value
#>   <chr>     <dbl>
#> 1 accuracy      1
#> 2 precision     1
#> 3 recall        1
#> 4 f1            1
# }
```
