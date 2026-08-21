# Test for significant interactions between variables

Test for significant interactions between variables

## Usage

``` r
tl_test_interactions(
  data,
  formula,
  var1 = NULL,
  var2 = NULL,
  all_pairs = FALSE,
  categorical_only = FALSE,
  numeric_only = FALSE,
  mixed_only = FALSE,
  alpha = 0.05
)
```

## Arguments

- data:

  A data frame containing the data

- formula:

  A formula specifying the base model without interactions

- var1:

  First variable to test for interactions

- var2:

  Second variable to test for interactions (if NULL, tests var1 with all
  others)

- all_pairs:

  Logical; whether to test all variable pairs

- categorical_only:

  Logical; whether to only test categorical variables

- numeric_only:

  Logical; whether to only test numeric variables

- mixed_only:

  Logical; whether to only test numeric-categorical pairs

- alpha:

  Significance level for interaction tests

## Value

A data frame with one row per tested interaction pair, containing
columns `var1`, `var2`, `p_value`, `significant` (logical), `delta_r2`
(change in R-squared), and `f_statistic`, sorted by `p_value` ascending.

## Examples

``` r
# \donttest{
results <- tl_test_interactions(mtcars, mpg ~ wt + hp + cyl,
  var1 = "wt", var2 = "hp")
# }
```
