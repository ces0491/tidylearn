# Contributing to tidylearn

Thanks for your interest in contributing! A few conventions to be aware of before you open a PR.

## Don't bump the package version

Please **do not modify `Version:` in `DESCRIPTION`** in your PR. Version bumps are handled by the maintainer as part of the release process — only the maintainer knows whether the next release is going to CRAN, when, and which version number it should carry.

## How versioning works here

tidylearn follows the standard R package convention described in [R Packages (2e), §21.2](https://r-pkgs.org/lifecycle.html#sec-lifecycle-version):

- **Released versions** on CRAN use three components: `X.Y.Z`.
- **Development versions** — the state of the package between CRAN releases — append a fourth component of `.9000` to the last released version.

So while CRAN has `X.Y.Z`, the `main` branch reads `Version: X.Y.Z.9000` for the entire time we are working toward the next release. The `.9000` suffix marks a dev build. When the maintainer is ready to submit the next release to CRAN, the `.9000` is dropped and the number raised to the new release, which is submitted. `main` then moves to that release plus `.9000` for the next dev cycle.

You'll sometimes see `.9001`, `.9002`, etc. used to mark meaningful dev milestones, but `.9000` alone is the default and is fine for all ordinary PRs.

## NEWS.md

If your change is user-visible (new feature, bug fix, API change), add a bullet under the current dev-version heading in `NEWS.md` — for example:

```markdown
# tidylearn X.Y.Z.9000

* Your change here.
```

If there isn't yet a dev-version heading at the top of the file, you can add one; the maintainer will reconcile it at release time.

Internal refactors and pure perf optimizations don't strictly need a NEWS entry, but one is welcome if the behaviour or performance characteristics are worth flagging to users.

## Tests

- Run `devtools::test()` locally before opening a PR.
- Add tests for any new behaviour or bug fix — see `tests/testthat/` for the existing patterns.
- `devtools::check()` should pass with no new WARNINGs or ERRORs. NOTEs may be acceptable depending on the case.
- The `"deep"` tests need `keras3` and a Python with Keras 3. They skip where either is missing, so run them when you touch that code.

## Code style

Follow the surrounding code. The package leans on tidyverse conventions:

- `snake_case` for function and variable names. The exception is a name that
  mirrors an argument of a wrapped package — e.g. `n.trees` (`gbm`), `minPts`
  (`dbscan`), `B` (`cluster::clusGap`) — which keeps that package's spelling
  so the interface stays familiar. `.lintr` allows `dotted.case` and
  `camelCase` for this reason; `.lintr` is parsed as strict DCF and cannot
  carry a comment, hence this note.
- `<-` (not `=`) for assignment.
- Exported functions need roxygen documentation with `@param`, `@return`, `@examples`, and `@export`.

`lintr::lint_package()` should report no issues. Install the package first
(`R CMD INSTALL .` or `devtools::install()`) so `object_usage_linter` can
resolve the namespace — otherwise it reports spurious "no visible global
function definition" lints for the package's own internal functions.

## Opening the PR

- Keep PRs focused — one feature or one fix per PR where possible.
- Include a short description of what the change does and why.
- Link any related issue.

Thanks again!
