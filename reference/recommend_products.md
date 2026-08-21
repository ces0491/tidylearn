# Generate Product Recommendations

Get product recommendations based on basket contents

## Usage

``` r
recommend_products(rules_obj, basket, top_n = 5, min_confidence = 0.5)
```

## Arguments

- rules_obj:

  A tidy_apriori object

- basket:

  Character vector of items in current basket

- top_n:

  Number of recommendations to return (default: 5)

- min_confidence:

  Minimum confidence threshold (default: 0.5)

## Value

A tibble with columns `rhs` (recommended item), `confidence`, `lift`,
and `support`, sorted by lift in descending order.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  # The basket has to cover the whole left-hand side of a rule, so a
  # basket of very common items usually matches nothing above the
  # confidence floor
  recommend_products(res, basket = c("flour", "baking powder"))
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
#> # A tibble: 2 × 4
#>   rhs          confidence  lift support
#>   <chr>             <dbl> <dbl>   <dbl>
#> 1 {sugar}           0.556 16.4  0.00102
#> 2 {whole milk}      0.523  2.05 0.00925
# }
```
