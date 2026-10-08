# Run a tidylearn pipeline

Run a tidylearn pipeline

## Usage

``` r
tl_run_pipeline(pipeline, verbose = TRUE)
```

## Arguments

- pipeline:

  A tidylearn pipeline object

- verbose:

  Logical; whether to print progress

## Value

The input `tidylearn_pipeline` object with its `$results` component
populated. Results include `$processed_data` (the training data after
preprocessing), `$preprocessing_stats` (the medians, modes, centres and
scales learned from the training data, replayed by
[`tl_predict_pipeline`](https://tidylearn.sheetsolved.com/reference/tl_predict_pipeline.md)),
`$model_results` (a named list of per-model fits and metrics),
`$best_model_name`, `$best_model` (the winning `tidylearn_model`), and
`$metric_values`.

The training data is every row when `validation = "cv"`: each model is
scored across the folds and then refitted on all of them. With
`validation = "split"` it is the training split alone, since the models
kept are the ones fitted there and scored on the test rows.

## Examples

``` r
# \donttest{
pipe <- tl_pipeline(iris, Species ~ .,
  models = list(tree = list(method = "tree")),
  evaluation = list(metrics = "accuracy", validation = "cv",
    cv_folds = 2, best_metric = "accuracy"))
pipe <- tl_run_pipeline(pipe, verbose = FALSE)
# }
```
