# Convert Association Rules to Tidy Tibble

Convert Association Rules to Tidy Tibble

## Usage

``` r
tidy_rules(rules)
```

## Arguments

- rules:

  A rules object from arules

## Value

A tibble with columns `rule_id`, `lhs`, `rhs`, and quality measures
(e.g., `support`, `confidence`, `lift`).

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  rules_obj <- arules::apriori(Groceries,
    parameter = list(supp = 0.001, conf = 0.5))
  rules_tbl <- tidy_rules(rules_obj)
}
#> Apriori
#> 
#> Parameter specification:
#>  confidence minval smax arem  aval originalSupport maxtime support minlen
#>         0.5    0.1    1 none FALSE            TRUE       5   0.001      1
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
# }
```
