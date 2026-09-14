# Print auto ML results

Print auto ML results

## Usage

``` r
# S3 method for class 'tidylearn_automl'
print(x, ...)
```

## Arguments

- x:

  A tidylearn_automl object

- ...:

  Additional arguments (ignored)

## Value

The input object `x`, returned invisibly.

## Examples

``` r
# \donttest{
result <- tl_auto_ml(iris, Species ~ .,
  time_budget = 10,
  use_reduction = FALSE,
  use_clustering = FALSE,
  cv_folds = 2)
#> Starting Auto ML with task: classification
#> Time budget: 10 seconds
#> 
#> [1/4] Training baseline models...
#>   Training: baseline_tree
#> 
#> [4/4] Training advanced models...
#> 
#> [*] Creating leaderboard...
#> 
#> Auto ML complete in 0.03 seconds
#> Best model: baseline_tree

# The leaderboard, the winner and the metric it was ranked on
print(result)
#> tidylearn Auto ML Results
#> =========================
#> Task: classification 
#> Metric: accuracy 
#> Runtime: 0.03 seconds
#> Models trained: 1 
#> 
#> Leaderboard:
#> # A tibble: 1 × 3
#>   model         score evaluation
#>   <chr>         <dbl> <chr>     
#> 1 baseline_tree 0.953 cv        
#> 
#> Best model: baseline_tree 
#> Best score: 0.9533333 
# }
```
