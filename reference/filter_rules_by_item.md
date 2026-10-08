# Filter Rules by Item

Subset rules containing specific items

## Usage

``` r
filter_rules_by_item(rules_obj, item, where = "both")
```

## Arguments

- rules_obj:

  A tidy_apriori object, an arules rules object, or a tibble of rules
  from
  [`tidy_rules`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md),
  which must keep its `lhs_items` and `rhs_items` columns

- item:

  Character; one item name, matched against whole items, so "coffee"
  does not match "instant coffee"

- where:

  Character; "lhs", "rhs", or "both" (default: "both")

## Value

A tibble of rules containing the specified `item` in the requested
position.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  filter_rules_by_item(res, "whole milk", where = "rhs")
}
#> # A tibble: 2,679 × 10
#>    rule_id lhs           rhs   support confidence coverage  lift count lhs_items
#>      <int> <chr>         <chr>   <dbl>      <dbl>    <dbl> <dbl> <int> <list>   
#>  1       1 {honey}       {who… 0.00112      0.733  0.00153  2.87    11 <chr [1]>
#>  2       3 {cocoa drink… {who… 0.00132      0.591  0.00224  2.31    13 <chr [1]>
#>  3       4 {pudding pow… {who… 0.00132      0.565  0.00234  2.21    13 <chr [1]>
#>  4       5 {cooking cho… {who… 0.00132      0.52   0.00254  2.04    13 <chr [1]>
#>  5       6 {cereals}     {who… 0.00366      0.643  0.00569  2.52    36 <chr [1]>
#>  6       7 {jam}         {who… 0.00295      0.547  0.00539  2.14    29 <chr [1]>
#>  7      10 {rice}        {who… 0.00468      0.613  0.00763  2.40    46 <chr [1]>
#>  8      11 {baking powd… {who… 0.00925      0.523  0.0177   2.05    91 <chr [1]>
#>  9      12 {liver loaf,… {who… 0.00102      0.667  0.00153  2.61    10 <chr [2]>
#> 10      14 {curd cheese… {who… 0.00102      0.625  0.00163  2.45    10 <chr [2]>
#> # ℹ 2,669 more rows
#> # ℹ 1 more variable: rhs_items <list>
# }
```
