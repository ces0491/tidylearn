# Standardize Data

Center and/or scale numeric variables

## Usage

``` r
standardize_data(data, center = TRUE, scale = TRUE)
```

## Arguments

- data:

  A data frame or tibble

- center:

  Logical; center variables? (default: TRUE)

- scale:

  Logical; scale variables to unit variance? (default: TRUE)

## Value

A tibble with numeric variables centered and/or scaled as specified;
non-numeric columns are returned unchanged. A grouped tibble is
standardised within each group, as
[`dplyr::mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
works on it; a rowwise tibble is standardised over its whole columns,
since a single value has no spread. Grouping and rowwise identifier
columns are left as they are.

## Examples

``` r
# \donttest{
std <- standardize_data(iris[, 1:4])
# }
```
