# Read from GitHub

Downloads a raw file from a GitHub repository and reads it into a
`tidylearn_data` object. Accepts either a full GitHub URL or a
`owner/repo` shorthand with a file path.

## Usage

``` r
tl_read_github(source, path = NULL, ref = "main", ..., trust_rds = FALSE)
```

## Arguments

- source:

  A GitHub URL or `"owner/repo"` string. A URL is either a file page
  (`https://github.com/<owner>/<repo>/blob/<ref>/<path>`, with `raw` in
  place of `blob` or with neither, and with or without `www.` and a
  query such as `?raw=true`) or a raw file
  (`https://raw.githubusercontent.com/...`), whose query is kept for the
  download. Zip archives are not read from GitHub: download the file and
  read it with
  [`tl_read_zip()`](https://tidylearn.sheetsolved.com/reference/tl_read_zip.md).

- path:

  Path to the file within the repository (required when `source` is
  `"owner/repo"` format).

- ref:

  Branch, tag, or commit SHA. Default is `"main"`.

- ...:

  Additional arguments passed to the format-specific reader.

- trust_rds:

  Logical. Read an `.rds`, `.rdata` or `.rda` file on R older than
  4.4.0? Those versions can run code embedded in a crafted file as it is
  read (CVE-2024-27322), so such files are refused there unless this is
  `TRUE`. It has no effect on R 4.4.0 or later. Default `FALSE`.

## Value

A `tidylearn_data` object containing the downloaded data.

## Reading R serialisation files

An `.rds` or `.rdata` file is rebuilt with
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) or
[`load()`](https://rdrr.io/r/base/load.html), which recreate whatever R
objects the file describes. Read them only from a repository you trust,
on any version of R.

## Examples

``` r
if (FALSE) { # \dontrun{
# Downloads over the network
tl_read_github("user/repo", path = "data/file.csv")
tl_read_github("https://github.com/user/repo/blob/main/data/file.csv")
} # }
```
