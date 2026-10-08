# Tidy Apriori Algorithm

Mine association rules using the Apriori algorithm with tidy output

## Usage

``` r
tidy_apriori(
  transactions,
  support = 0.01,
  confidence = 0.5,
  minlen = 2,
  maxlen = 10,
  target = "rules",
  control = list(verbose = FALSE),
  ...
)
```

## Arguments

- transactions:

  A transactions object or data frame

- support:

  Minimum support (default: 0.01)

- confidence:

  Minimum confidence (default: 0.5)

- minlen:

  Minimum rule length (default: 2)

- maxlen:

  Maximum rule length (default: 10)

- target:

  Type of association mined: "rules" (default), "frequent itemsets",
  "maximally frequent itemsets"

- control:

  A list of algorithmic controls for
  [`apriori`](https://rdrr.io/pkg/arules/man/apriori.html). The mining
  trace is off unless the list sets `verbose = TRUE`; any other entries
  are passed as given.

- ...:

  Further arguments passed to
  [`apriori`](https://rdrr.io/pkg/arules/man/apriori.html):
  `appearance`, to restrict where items may appear, or more mining
  parameters, such as `smax` or `maxtime`, which it adds to the ones
  above.

## Value

A list of class "tidy_apriori" containing:

- rules_tbl: tibble of rules, as
  [`tidy_rules`](https://tidylearn.sheetsolved.com/reference/tidy_rules.md)
  returns it, or `NULL` for an itemset target

- rules: original arules object, rules or itemsets

- parameters: parameters used

- n_rules: number of rules, or of itemsets for an itemset target

- itemsets_tbl: for an itemset target only, a tibble with `itemset_id`,
  `itemset` (its label), `size`, the quality measures, and `items`, a
  list column holding each itemset's items

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
data("Groceries", package = "arules")

# Basic apriori
rules <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)

# Access rules
rules$rules_tbl
}
#> # A tibble: 5,668 × 10
#>    rule_id lhs           rhs   support confidence coverage  lift count lhs_items
#>      <int> <chr>         <chr>   <dbl>      <dbl>    <dbl> <dbl> <int> <list>   
#>  1       1 {honey}       {who… 0.00112      0.733  0.00153  2.87    11 <chr [1]>
#>  2       2 {tidbits}     {rol… 0.00122      0.522  0.00234  2.84    12 <chr [1]>
#>  3       3 {cocoa drink… {who… 0.00132      0.591  0.00224  2.31    13 <chr [1]>
#>  4       4 {pudding pow… {who… 0.00132      0.565  0.00234  2.21    13 <chr [1]>
#>  5       5 {cooking cho… {who… 0.00132      0.52   0.00254  2.04    13 <chr [1]>
#>  6       6 {cereals}     {who… 0.00366      0.643  0.00569  2.52    36 <chr [1]>
#>  7       7 {jam}         {who… 0.00295      0.547  0.00539  2.14    29 <chr [1]>
#>  8       8 {specialty c… {oth… 0.00427      0.5    0.00854  2.58    42 <chr [1]>
#>  9       9 {rice}        {oth… 0.00397      0.52   0.00763  2.69    39 <chr [1]>
#> 10      10 {rice}        {who… 0.00468      0.613  0.00763  2.40    46 <chr [1]>
#> # ℹ 5,658 more rows
#> # ℹ 1 more variable: rhs_items <list>
# }
```
