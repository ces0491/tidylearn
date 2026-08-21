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
  target = "rules"
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

## Value

A list of class "tidy_rules" containing:

- rules_tbl: tibble of rules with lhs, rhs, and quality measures

- rules: original rules object

- parameters: parameters used

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
#> # A tibble: 5,668 × 8
#>    rule_id lhs                 rhs       support confidence coverage  lift count
#>      <int> <chr>               <chr>       <dbl>      <dbl>    <dbl> <dbl> <int>
#>  1       1 {honey}             {whole m… 0.00112      0.733  0.00153  2.87    11
#>  2       2 {tidbits}           {rolls/b… 0.00122      0.522  0.00234  2.84    12
#>  3       3 {cocoa drinks}      {whole m… 0.00132      0.591  0.00224  2.31    13
#>  4       4 {pudding powder}    {whole m… 0.00132      0.565  0.00234  2.21    13
#>  5       5 {cooking chocolate} {whole m… 0.00132      0.52   0.00254  2.04    13
#>  6       6 {cereals}           {whole m… 0.00366      0.643  0.00569  2.52    36
#>  7       7 {jam}               {whole m… 0.00295      0.547  0.00539  2.14    29
#>  8       8 {specialty cheese}  {other v… 0.00427      0.5    0.00854  2.58    42
#>  9       9 {rice}              {other v… 0.00397      0.52   0.00763  2.69    39
#> 10      10 {rice}              {whole m… 0.00468      0.613  0.00763  2.40    46
#> # ℹ 5,658 more rows
# }
```
