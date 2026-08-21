# Positive-class argument for yardstick binary metrics

tidylearn's positive class is the second factor level. yardstick's
default is the first, and its `event_level` argument only applies to the
binary case – passing it for a multiclass problem warns.

## Usage

``` r
tl_event_level_args(actuals)
```

## Arguments

- actuals:

  A factor of ground-truth values

## Value

A list to splice into a yardstick call: `event_level = "second"` for a
two-level factor, empty otherwise
