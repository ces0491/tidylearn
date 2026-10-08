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

  Additional arguments passed to the
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) method keras
  provides for its models, such as `to_file` or `dpi`. `show_shapes` and
  `show_layer_names` default to `TRUE`.

## Value

`NULL`, invisibly. Called for its side effect: keras draws the
architecture diagram on the current graphics device, or writes it to
`to_file`. keras renders it through the Python packages `pydot` and
`graphviz`, and errors saying so when they are not installed.

## Examples

``` r
if (FALSE) { # \dontrun{
if (requireNamespace("keras", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "deep", epochs = 5)
  tl_plot_deep_architecture(model)
}
} # }
```
