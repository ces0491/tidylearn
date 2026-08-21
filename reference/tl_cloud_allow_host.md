# Allow an additional host for cloud uploads in this R session

tidylearn uploads training data only to Modal's own hosts
(`*.modal.run`, `*.modal.com`). Modal customers serving Web Functions
from a custom domain can add that domain here.

## Usage

``` r
tl_cloud_allow_host(host)
```

## Arguments

- host:

  A character vector of host names to allow, or `NULL` to clear every
  host added this session. Bare host names only — not URLs, ports, paths
  or wildcards.

## Value

The full allowlist after the change, invisibly.

## Details

This widens the set of destinations your data may be sent to, so it is
deliberately a per-session call rather than an option or an environment
variable: a shared `.Rprofile` or an inherited environment should not be
able to add a destination without you writing the call. Additions are
never persisted and are forgotten when the session ends.

Hosts match themselves and their subdomains. Adding `"fits.example.com"`
accepts `https://fits.example.com` and `https://a.fits.example.com`, and
nothing else.

## See also

[`tl_cloud_allowed_hosts()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allowed_hosts.md),
and T9 in
`system.file("security/threat-model.md", package = "tidylearn")`.

## Examples

``` r
tl_cloud_allow_host("fits.example.com")
#> Cloud uploads may now also go to: fits.example.com. This is in addition to Modal's own hosts and lasts for this R session only.
tl_cloud_allowed_hosts()
#> [1] "modal.run"        "modal.com"        "fits.example.com"
tl_cloud_allow_host(NULL)
#> Cleared all additional cloud hosts for this R session.
```
