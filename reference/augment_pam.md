# Augment Data with PAM Cluster Assignments

Augment Data with PAM Cluster Assignments

## Usage

``` r
augment_pam(pam_obj, data)
```

## Arguments

- pam_obj:

  A tidy_pam object

- data:

  Original data frame

## Value

A tibble containing the original `data` with an additional `cluster`
factor column indicating cluster assignments.

## Examples

``` r
# \donttest{
pm <- tidy_pam(iris[, 1:4], k = 3)
augmented <- augment_pam(pm, iris)
# }
```
