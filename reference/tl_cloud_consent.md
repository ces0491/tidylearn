# Grant or revoke cloud upload consent for this R session

Fitting with `compute = "cloud"` uploads your training data to your own
Modal account, which is a third party. tidylearn will not do that
without explicit consent on every call.

## Usage

``` r
tl_cloud_consent(consent = TRUE)
```

## Arguments

- consent:

  `TRUE` to allow cloud uploads for the rest of the session, `FALSE` to
  revoke.

## Value

The previous consent state, invisibly.

## Details

There are two ways to give it. Pass `confirm_upload = TRUE` to each
[`tl_model()`](https://tidylearn.sheetsolved.com/reference/tl_model.md)
call, or call `tl_cloud_consent()` once to opt in for the rest of the
session. The session lock exists for batch and non-interactive work,
where a per-call argument is repetitive.

The lock is **not** persisted. It is forgotten when the R session ends,
and it is never written to disk. Revoke it early with
`tl_cloud_consent(FALSE)`.

tidylearn never prompts interactively for consent, so scripts, CI and
`Rscript` behave the same as an interactive session.

## See also

The full contract is in
`system.file("security/threat-model.md", package = "tidylearn")`.

## Examples

``` r
# Opt in for the session, then revoke.
old <- tl_cloud_consent(TRUE)
#> Cloud uploads enabled for this R session. Training data passed to tl_model(compute = 'cloud') will be sent to your Modal account. Revoke with tl_cloud_consent(FALSE).
tl_cloud_consent(FALSE)
#> Cloud uploads disabled for this R session.
```
