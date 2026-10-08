# Summarize Association Rules

Get summary statistics about rules

## Usage

``` r
summarize_rules(rules_obj)
```

## Arguments

- rules_obj:

  A tidy_apriori object, an arules rules object, or a rules tibble

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
