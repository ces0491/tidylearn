# Summarize Association Rules

Get summary statistics about rules

## Usage

``` r
summarize_rules(rules_obj)
```

## Arguments

- rules_obj:

  A tidy_apriori object or rules tibble

## Value

A list with `n_rules` and summary statistics (`min`, `max`, `mean`,
`median`) for `support`, `confidence`, and `lift`.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  summarize_rules(res)
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
#> $n_rules
#> [1] 5668
#> 
#> $support
#> $support$min
#> [1] 0.001016777
#> 
#> $support$max
#> [1] 0.02226741
#> 
#> $support$mean
#> [1] 0.001667797
#> 
#> $support$median
#> [1] 0.00132181
#> 
#> 
#> $confidence
#> $confidence$min
#> [1] 0.5
#> 
#> $confidence$max
#> [1] 1
#> 
#> $confidence$mean
#> [1] 0.6249694
#> 
#> $confidence$median
#> [1] 0.6
#> 
#> 
#> $lift
#> $lift$min
#> [1] 1.956825
#> 
#> $lift$max
#> [1] 18.99565
#> 
#> $lift$mean
#> [1] 3.262302
#> 
#> $lift$median
#> [1] 2.898999
#> 
#> 
# }
```
