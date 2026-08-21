# Hosts tidylearn will currently upload to

Modal's own hosts, plus anything added with
[`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)
during this session.

## Usage

``` r
tl_cloud_allowed_hosts()
```

## Value

A character vector of host names.

## See also

[`tl_cloud_allow_host()`](https://tidylearn.sheetsolved.com/reference/tl_cloud_allow_host.md)

## Examples

``` r
tl_cloud_allowed_hosts()
#> [1] "modal.run" "modal.com"
```
