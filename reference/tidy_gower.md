# Gower Distance Calculation

Computes Gower distance for mixed data types (numeric, factor, ordered)

## Usage

``` r
tidy_gower(data, weights = NULL)
```

## Arguments

- data:

  A data frame or tibble

- weights:

  Optional named vector of variable weights (default: equal weights)

## Value

A [`dist`](https://rdrr.io/r/stats/dist.html) object containing Gower
distances, with the `method` attribute set to `"gower"`.

## Details

Gower distance handles mixed data types:

- Numeric: range-normalized Manhattan distance

- Factor/Character: 0 if same, 1 if different

- Ordered: treated as numeric ranks

Formula: d_ij = sum(w_k \* d_ijk) / sum(w_k) where d_ijk is the
dissimilarity for variable k between obs i and j

## Examples

``` r
# Create example data with mixed types
car_data <- data.frame(
  horsepower = c(130, 250, 180),
  weight = c(1200, 1650, 1420),
  color = factor(c("red", "black", "blue"))
)

# Compute Gower distance
gower_dist <- tidy_gower(car_data)
```
