# Find important interactions automatically

Find important interactions automatically

## Usage

``` r
tl_auto_interactions(
  data,
  formula,
  top_n = 3,
  min_r2_change = 0.01,
  max_p_value = 0.05,
  exclude_vars = NULL
)
```

## Arguments

- data:

  A data frame containing the data

- formula:

  A formula specifying the base model without interactions

- top_n:

  Number of top interactions to return

- min_r2_change:

  Minimum change in R-squared to consider

- max_p_value:

  Maximum p-value for significance

- exclude_vars:

  Character vector of variables to exclude from interaction testing

## Value

A tidylearn model object (class `"tidylearn_model"`) fitted with the top
significant interaction terms added to the formula. The interaction test
results and selected interactions are stored as attributes
`"interaction_tests"` and `"selected_interactions"`.

## Examples

``` r
# \donttest{
model <- tl_auto_interactions(mtcars, mpg ~ wt + hp + cyl, top_n = 2)
# }
```
