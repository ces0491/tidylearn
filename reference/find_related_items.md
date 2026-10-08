# Find Related Items

Find items frequently purchased with a given item

## Usage

``` r
find_related_items(rules_obj, item, min_lift = 1.5, top_n = 10)
```

## Arguments

- rules_obj:

  A tidy_apriori object, an arules rules object, or a tibble of rules
  from
  [`tidy_rules`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md),
  which must keep its `lhs_items` and `rhs_items` columns

- item:

  Character; one item name to find associations for, matched against
  whole items

- min_lift:

  Minimum lift threshold (default: 1.5)

- top_n:

  Number of top associations to return (default: 10)

## Value

A tibble of rules involving the specified `item`, filtered by `min_lift`
and sorted by lift in descending order.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  find_related_items(res, "whole milk", min_lift = 1.5)
}
#> # A tibble: 10 × 10
#>    rule_id lhs           rhs   support confidence coverage  lift count lhs_items
#>      <int> <chr>         <chr>   <dbl>      <dbl>    <dbl> <dbl> <int> <list>   
#>  1      55 {whole milk,… {ham… 0.00153      0.5    0.00305 15.0     15 <chr [2]>
#>  2    5638 {tropical fr… {but… 0.00102      0.625  0.00163 11.3     10 <chr [5]>
#>  3    5633 {tropical fr… {bee… 0.00112      0.55   0.00203 10.5     11 <chr [5]>
#>  4    4734 {tropical fr… {but… 0.00102      0.556  0.00183 10.0     10 <chr [4]>
#>  5    1827 {whole milk,… {but… 0.00142      0.538  0.00264  9.72    14 <chr [3]>
#>  6    1826 {whole milk,… {whi… 0.00142      0.667  0.00214  9.30    14 <chr [3]>
#>  7    4820 {citrus frui… {dom… 0.00112      0.579  0.00193  9.12    11 <chr [4]>
#>  8    4810 {whole milk,… {whi… 0.00112      0.647  0.00173  9.03    11 <chr [4]>
#>  9    5044 {other veget… {but… 0.00102      0.5    0.00203  9.02    10 <chr [4]>
#> 10    2699 {citrus frui… {dom… 0.00163      0.571  0.00285  9.01    16 <chr [3]>
#> # ℹ 1 more variable: rhs_items <list>
# }
```
