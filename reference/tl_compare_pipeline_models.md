# Compare models from a pipeline

Compare models from a pipeline

## Usage

``` r
tl_compare_pipeline_models(pipeline, metrics = NULL)
```

## Arguments

- pipeline:

  A tidylearn pipeline object with results

- metrics:

  Character vector of metrics to compare (if NULL, uses all available)

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) object
showing a faceted bar chart comparing metric values across models, with
the best model highlighted.
