# Find Related Items

Find items frequently purchased with a given item

## Usage

``` r
find_related_items(rules_obj, item, min_lift = 1.5, top_n = 10)
```

## Arguments

- rules_obj:

  A tidy_apriori object

- item:

  Character; item to find associations for

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
#> # A tibble: 10 × 8
#>    rule_id lhs                     rhs   support confidence coverage  lift count
#>      <int> <chr>                   <chr>   <dbl>      <dbl>    <dbl> <dbl> <int>
#>  1      55 {whole milk,Instant fo… {ham… 0.00153      0.5    0.00305 15.0     15
#>  2    5638 {tropical fruit,other … {but… 0.00102      0.625  0.00163 11.3     10
#>  3    5633 {tropical fruit,root v… {bee… 0.00112      0.55   0.00203 10.5     11
#>  4    4734 {tropical fruit,whole … {but… 0.00102      0.556  0.00183 10.0     10
#>  5    1827 {whole milk,whipped/so… {but… 0.00142      0.538  0.00264  9.72    14
#>  6    1826 {whole milk,butter,har… {whi… 0.00142      0.667  0.00214  9.30    14
#>  7    4820 {citrus fruit,other ve… {dom… 0.00112      0.579  0.00193  9.12    11
#>  8    4810 {whole milk,curd,yogur… {whi… 0.00112      0.647  0.00173  9.03    11
#>  9    5044 {other vegetables,whol… {but… 0.00102      0.5    0.00203  9.02    10
#> 10    2699 {citrus fruit,whole mi… {dom… 0.00163      0.571  0.00285  9.01    16
# }
```
