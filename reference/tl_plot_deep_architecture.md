# Plot deep learning model architecture

Plot deep learning model architecture

## Usage

``` r
tl_plot_deep_architecture(model, ...)
```

## Arguments

- model:

  A tidylearn deep learning model object

- ...:

  Additional arguments

## Value

The return value of `keras::plot_model()`, an architecture diagram of
the Keras model.

## Examples

``` r
if (FALSE) { # \dontrun{
if (requireNamespace("keras", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 5)
  tl_plot_deep_architecture(model)
}
} # }
```
