# Calculate the area under the precision-recall curve

Reads the area off a curve ROCR built, and agrees with
[`yardstick::pr_auc()`](https://yardstick.tidymodels.org/reference/pr_auc.html).
ROCR's curve starts at recall 0, where nothing is yet called positive
and precision is 0/0. The area has to start there too, at precision 1 as
yardstick's does: integrating from the first finite point lost
everything before it, so a perfect ranking of 5 positives in 20 scored
0.8, a tree on two iris species 0.07, and constant scores left no area
at all.

## Usage

``` r
tl_calculate_pr_auc(perf)
```

## Arguments

- perf:

  A ROCR performance object of precision against recall

## Value

The area under the precision-recall curve, or `NA` when the curve has
fewer than two points
