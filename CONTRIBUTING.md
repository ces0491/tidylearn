# Contributing to tidylearn

Thanks for your interest in contributing! A few conventions to be aware of before you open a PR.

## Don't bump the package version

Please **do not modify `Version:` in `DESCRIPTION`** in your PR. Version bumps are handled by the maintainer as part of the release process — only the maintainer knows whether the next release is going to CRAN, when, and which version number it should carry.

## How versioning works here

tidylearn follows the standard R package convention described in [R Packages (2e), §21.2](https://r-pkgs.org/lifecycle.html#sec-lifecycle-version):

- **Released versions** on CRAN use three components: `X.Y.Z` (e.g. `0.3.0`).
- **Development versions** — the state of the package between CRAN releases — append a fourth component of `.9000` to the last released version.

So if CRAN currently has `0.3.0`, the `main` branch will read `Version: 0.3.0.9000` for the entire time we're working toward the next release. The `.9000` suffix is a signal that says "this is a dev build, not a CRAN release." When the maintainer is ready to submit the next release to CRAN, the version is bumped to `0.3.1` (dropping `.9000`), submitted, and then immediately bumped back to `0.3.1.9000` on `main` for the next dev cycle.

You'll sometimes see `.9001`, `.9002`, etc. used to mark meaningful dev milestones, but `.9000` alone is the default and is fine for all ordinary PRs.

## NEWS.md

If your change is user-visible (new feature, bug fix, API change), add a bullet under the current dev-version heading in `NEWS.md` — for example:

```markdown
# tidylearn 0.3.0.9000

* Your change here.
```

If there isn't yet a dev-version heading at the top of the file, you can add one; the maintainer will reconcile it at release time.

Internal refactors and pure perf optimizations don't strictly need a NEWS entry, but one is welcome if the behaviour or performance characteristics are worth flagging to users.

## Tests

- Run `devtools::test()` locally before opening a PR.
- Add tests for any new behaviour or bug fix — see `tests/testthat/` for the existing patterns.
- `devtools::check()` should pass with no new WARNINGs or ERRORs. NOTEs may be acceptable depending on the case.

## Code style

Follow the surrounding code. The package leans on tidyverse conventions:

- `snake_case` for function and variable names.
- `<-` (not `=`) for assignment.
- Exported functions need roxygen documentation with `@param`, `@return`, `@examples`, and `@export`.

## Opening the PR

- Keep PRs focused — one feature or one fix per PR where possible.
- Include a short description of what the change does and why.
- Link any related issue.

Thanks again!
