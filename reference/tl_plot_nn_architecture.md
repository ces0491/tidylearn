# Plot neural network architecture

Plot neural network architecture

## Usage

``` r
tl_plot_nn_architecture(model, ...)
```

## Arguments

- model:

  A tidylearn neural network model object

- ...:

  Additional arguments

## Value

The return value of
[`plotnet`](https://rdrr.io/pkg/NeuralNetTools/man/plotnet.html), called
for its side effect of drawing the network diagram, or `NULL` if the
NeuralNetTools package is not installed.

## Examples

``` r
# \donttest{
if (requireNamespace("NeuralNetTools", quietly = TRUE)) {
  model <- tl_model(iris, Species ~ ., method = "nn", size = 3)
  tl_plot_nn_architecture(model)
}

# }
```
