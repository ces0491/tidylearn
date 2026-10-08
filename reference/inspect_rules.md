# Inspect Association Rules

View rules sorted by various quality measures

## Usage

``` r
inspect_rules(rules_obj, by = "lift", n = 10, decreasing = TRUE)
```

## Arguments

- rules_obj:

  A tidy_apriori object, an arules rules or itemsets object, or a tibble
  of rules

- by:

  Sort by: "support", "confidence", "lift" (default), "count". Itemsets
  have no lift, so for them the default sorts by support.

- n:

  Number of rules to display (default: 10)

- decreasing:

  If TRUE (default), the `n` rules with the highest values of `by`,
  highest first; if FALSE, the `n` with the lowest, lowest first, as in
  arules' `head(by = )`.

## Value

A tibble of the `n` rules ranked highest (or, with `decreasing = FALSE`,
lowest) by the quality measure `by`.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  inspect_rules(res, by = "lift", n = 5)
}
#> # A tibble: 5 × 10
#>   rule_id lhs            rhs   support confidence coverage  lift count lhs_items
#>     <int> <chr>          <chr>   <dbl>      <dbl>    <dbl> <dbl> <int> <list>   
#> 1      53 {Instant food… {ham… 0.00122      0.632  0.00193  19.0    12 <chr [2]>
#> 2      37 {soda,popcorn} {sal… 0.00122      0.632  0.00193  16.7    12 <chr [2]>
#> 3     444 {flour,baking… {sug… 0.00102      0.556  0.00183  16.4    10 <chr [2]>
#> 4     327 {ham,processe… {whi… 0.00193      0.633  0.00305  15.0    19 <chr [2]>
#> 5      55 {whole milk,I… {ham… 0.00153      0.5    0.00305  15.0    15 <chr [2]>
#> # ℹ 1 more variable: rhs_items <list>
# }
```
