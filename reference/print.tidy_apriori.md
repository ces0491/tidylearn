# Print Method for tidy_apriori

Print Method for tidy_apriori

## Usage

``` r
# S3 method for class 'tidy_apriori'
print(x, ...)
```

## Arguments

- x:

  A tidy_apriori object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
if (requireNamespace("arules", quietly = TRUE)) {
  data("Groceries", package = "arules")
  res <- tidy_apriori(Groceries, support = 0.001, confidence = 0.5)
  print(res)
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
#> Tidy Apriori Results
#> ====================
#> 
#> Parameters:
#>   Minimum support:    0.001 
#>   Minimum confidence: 0.5 
#>   Rule length:        2 - 10 
#> 
#> Results:
#>   Number of rules: 5668 
#> 
#> Quality Measure Summary:
#>   Support:     0.0010 - 0.0223 (mean: 0.0017) 
#>   Confidence:  0.5000 - 1.0000 (mean: 0.6250) 
#>   Lift:        1.96 - 19.00 (mean: 3.26) 
#> 
#> Top 5 rules by lift:
#> # A tibble: 5 × 8
#>   rule_id lhs                      rhs   support confidence coverage  lift count
#>     <int> <chr>                    <chr>   <dbl>      <dbl>    <dbl> <dbl> <int>
#> 1      53 {Instant food products,… {ham… 0.00122      0.632  0.00193  19.0    12
#> 2      37 {soda,popcorn}           {sal… 0.00122      0.632  0.00193  16.7    12
#> 3     444 {flour,baking powder}    {sug… 0.00102      0.556  0.00183  16.4    10
#> 4     327 {ham,processed cheese}   {whi… 0.00193      0.633  0.00305  15.0    19
#> 5      55 {whole milk,Instant foo… {ham… 0.00153      0.5    0.00305  15.0    15
#> 
#> Use inspect_rules() to view more rules
#> Use visualize_rules() to create visualizations
# }
```
