# Cloud jobs submitted in this R session

Every cloud submission is recorded here so that no job runs invisibly. A
job disappears from this list when its result is collected or it is
cancelled.

## Usage

``` r
tl_cloud_jobs()
```

## Value

A tibble with one row per in-flight job: `call_id`, `method`,
`submitted_at`, `timeout_seconds` and `worst_case_cost`. Zero rows when
nothing is in flight.

## Details

A submitted job runs on Modal whatever this R session does. If the
session ends while a job is still listed here, that job keeps running
and keeps billing until it finishes or its timeout kills it — this list
cannot survive the session, but the timeout does.

## Examples

``` r
# Nothing in flight in a fresh session
tl_cloud_jobs()
#> # A tibble: 0 × 5
#> # ℹ 5 variables: call_id <chr>, method <chr>, submitted_at <dttm>,
#> #   timeout_seconds <int>, worst_case_cost <dbl>
```
