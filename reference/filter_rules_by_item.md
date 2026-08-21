# Filter Rules by Item

Subset rules containing specific items

## Usage

``` r
filter_rules_by_item(rules_obj, item, where = "both")
```

## Arguments

- rules_obj:

  A tidy_apriori object or tibble of rules

- item:

  Character; item to filter by

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
#> Apriori
#> 
#> Parameter specification:
#>  confidence minval smax arem  aval originalSupport maxtime support minlen
#>         0.5    0.1    1 none FALSE            TRUE       5   0.001      2
#>  maxlen target  ext
#>      10  rules TRUE
#> 
#> Algorithmic control:
#>  filter tree heap memopt load sort verbose
#>     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
#> 
#> Absolute minimum support count: 9 
#> 
#> set item appearances ...[0 item(s)] done [0.00s].
#> set transactions ...[169 item(s), 9835 transaction(s)] done [0.00s].
#> sorting and recoding items ... [157 item(s)] done [0.00s].
#> creating transaction tree ... done [0.00s].
#> checking subsets of size 1 2 3 4 5 6 done [0.01s].
#> writing ... [5668 rule(s)] done [0.00s].
#> creating S4 object  ... done [0.00s].
#> # A tibble: 2,679 × 8
#>    rule_id lhs                     rhs   support confidence coverage  lift count
#>      <int> <chr>                   <chr>   <dbl>      <dbl>    <dbl> <dbl> <int>
#>  1       1 {honey}                 {who… 0.00112      0.733  0.00153  2.87    11
#>  2       3 {cocoa drinks}          {who… 0.00132      0.591  0.00224  2.31    13
#>  3       4 {pudding powder}        {who… 0.00132      0.565  0.00234  2.21    13
#>  4       5 {cooking chocolate}     {who… 0.00132      0.52   0.00254  2.04    13
#>  5       6 {cereals}               {who… 0.00366      0.643  0.00569  2.52    36
#>  6       7 {jam}                   {who… 0.00295      0.547  0.00539  2.14    29
#>  7      10 {rice}                  {who… 0.00468      0.613  0.00763  2.40    46
#>  8      11 {baking powder}         {who… 0.00925      0.523  0.0177   2.05    91
#>  9      12 {liver loaf,yogurt}     {who… 0.00102      0.667  0.00153  2.61    10
#> 10      14 {curd cheese,rolls/bun… {who… 0.00102      0.625  0.00163  2.45    10
#> # ℹ 2,669 more rows
# }
```
