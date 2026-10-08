# Generate Product Recommendations

Get product recommendations based on basket contents

## Usage

``` r
recommend_products(rules_obj, basket, top_n = 5, min_confidence = 0.5)
```

## Arguments

- rules_obj:

  A tidy_apriori object, an arules rules object, or a tibble of rules
  from
  [`tidy_rules`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md),
  which must keep its `lhs_items` and `rhs_items` columns

- basket:

  Character vector of items in current basket

- top_n:

  Number of recommendations to return (default: 5)

- min_confidence:

  Minimum confidence threshold (default: 0.5)

## Value

A tibble with columns `rhs` (recommended item), `confidence`, `lift`,
and `support`, sorted by lift in descending order. A rule is used when
the basket holds its whole left-hand side and none of its right-hand
side, and each product is listed once, from its highest-lift rule.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  # The basket has to cover the whole left-hand side of a rule, so a
  # basket of very common items usually matches nothing above the
  # confidence floor
  recommend_products(res, basket = c("flour", "baking powder"))
}
#> # A tibble: 2 × 4
#>   rhs          confidence  lift support
#>   <chr>             <dbl> <dbl>   <dbl>
#> 1 {sugar}           0.556 16.4  0.00102
#> 2 {whole milk}      0.523  2.05 0.00925
# }
```
