# Read from GitHub

Downloads a raw file from a GitHub repository and reads it into a
`tidylearn_data` object. Accepts either a full GitHub URL or a
`owner/repo` shorthand with a file path.

## Usage

``` r
tl_read_github(source, path = NULL, ref = "main", ...)
```

## Arguments

- source:

  A GitHub URL or `"owner/repo"` string.

- path:

  Path to the file within the repository (required when `source` is
  `"owner/repo"` format).

- ref:

  Branch, tag, or commit SHA. Default is `"main"`.

- ...:

  Additional arguments passed to the format-specific reader.

## Value

A `tidylearn_data` object containing the downloaded data.

## Examples

``` r
# \donttest{
# data <- tl_read_github("user/repo", path = "data/file.csv")
# data <- tl_read_github(
#   "https://github.com/user/repo/blob/main/data/file.csv"
# )
# }
```
