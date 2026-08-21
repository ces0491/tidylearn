# tidylearn cloud compute — threat model

**Status:** DRAFT — describes the commitments tidylearn's cloud-compute
integration will meet when the Modal-backed `compute = "cloud"` path lands.
**Last updated:** 2026-08-06
**Scope:** the cloud-compute paths in tidylearn (`tl_compute_advisor()`'s
cloud-tier estimates, `tl_resolve_compute()`'s `"cloud"` branch, and the
forthcoming `tl_model(..., compute = "cloud")` submission to Modal).

**API names below are provisional.** `confirm_upload`, `dry_run` and
`tl_cloud_consent()` describe the intended design and do not exist yet;
`tl_model()` is the existing entry point they would attach to. The
commitments are binding, the spellings are not.

**Transport.** tidylearn talks to Modal over plain HTTPS from R, using
`httr2` against a Modal Web Function the user has deployed to their own
Modal account. Modal's control plane is gRPC with SDKs for Python, JS and
Go only; there is no R SDK and no general REST API for invoking Functions,
so the Web Function endpoint is the supported language-agnostic surface.
tidylearn does not embed a Python runtime and does not call Modal through
reticulate. The worker container runs R, so fitted models cross the wire as
native R serialisation rather than a per-method export format.

This document is the contract for what cloud-compute in tidylearn will and
will not do. Anyone reviewing the Modal-integration PR should be able to
read this doc and verify, via grep / code review / test runs, that each
mitigation actually holds in the implementation.

The doc is intentionally narrow. It does not cover Modal's own
infrastructure security (see Modal's security docs and SOC 2 report), nor
local OS / R session hardening, nor threats requiring physical access to
the user's machine, nor cryptanalysis of TLS.

---

## 1. Trust boundaries

```text
+-------------------------+                      +-----------------------+
| User's R session        |                      | Modal infrastructure  |
| (trusted)               |                      | (third-party)         |
|                         |                      |                       |
|  +-------------------+  |   HTTPS + proxy      |  +-----------------+  |
|  | tidylearn R code  |--+---token headers----->+->| Web Function    |  |
|  | (trusted)         |  |   to *.modal.run     |  | endpoint        |  |
|  |   via httr2       |<-+---model bytes--------+--+--------+--------+  |
|  +---------+---------+  |                      |           |           |
|            |            |                      |  +--------v--------+  |
|  +---------v---------+  |                      |  | ephemeral R     |  |
|  | local data        |  |                      |  | worker container|  |
|  | (sensitive,       |  |                      |  | (per job)       |  |
|  |  user-owned)      |  |                      |  +-----------------+  |
|  +-------------------+  |                      +-----------------------+
|                         |
|  +-------------------+  |   read at call time, never logged
|  | proxy token       |  |   (env var / user-supplied)
|  | wk-... / ws-...   |  |
|  +-------------------+  |
+-------------------------+

  Deployment (one-time, outside R): user runs `modal deploy` with Modal's
  Python CLI. The account API token stays with the CLI and never enters R.
```

Boundaries:

- **R session ↔ tidylearn code** — trusted in both directions. tidylearn
  runs in the user's R process with the user's privileges.
- **tidylearn code ↔ Modal endpoint** — tidylearn constructs the HTTPS
  request itself with `httr2`. There is no SDK sub-component inside the
  trusted boundary; the transport is tidylearn's own code and is
  therefore fully in scope for this document.
- **tidylearn ↔ Modal infrastructure** — TLS to `*.modal.run`. tidylearn
  does not implement its own cryptography and does not disable
  certificate verification.
- **User's local data → Modal** — sensitive data crossing a third-party
  boundary. This is the most consequential transition and is gated by
  explicit consent (see [Section 4](#4-data-egress-consent-ux)).
- **User's Modal account API token** — never crosses into tidylearn code.
  It is used only by Modal's Python CLI at deploy time, which runs outside
  R (see [T1](#t1-token-handling-in-tidylearn)).
- **User's Modal proxy token** — a distinct, endpoint-scoped credential
  (`wk-…` ID plus `ws-…` secret) that tidylearn *does* handle, because it
  must set the `Modal-Key` / `Modal-Secret` request headers itself. This
  is the significant change from the reticulate-based design and is
  covered by [T1](#t1-token-handling-in-tidylearn).

## 2. Assets

| Asset | Sensitivity | Storage | Lifetime |
| --- | --- | --- | --- |
| Modal account API token | High | `~/.modal.toml` (user OS), used only by the Python CLI at deploy time | Until revoked |
| Modal proxy token (`wk-` / `ws-`) | High — endpoint-scoped | User environment; read into the R process per request, never persisted | Until revoked |
| Training data (rows × cols) | User-determined; may contain PII | In-memory in R, transmitted to Modal during fit | Ephemeral on Modal — destroyed with container |
| Fitted model artifacts | Medium — may memorize training data | Returned to R session as raw bytes; not persisted on Modal. Keras models travel as a separate hdf5 payload, since they cannot cross a process boundary through base R serialisation | Lives in user's R session |
| Hyperparameters / model spec | Low | In-memory in R, transmitted to Modal | Ephemeral |
| Job stdout / logs | Low — should contain no row-level data | Modal log retention (per Modal's policy) | Per Modal |

## 3. Threats and mitigations

### T1: Token handling in tidylearn

**Scenario.** tidylearn causes a Modal credential to leak — by reading the
account token, by printing a token to the console, by including one in an
error message or an `httr2` verbose/debug dump, or by writing one into a
saved object.

**Risk.** High for the account API token: it can spin up arbitrary Modal
jobs at the user's expense and reach anything in their workspace. Lower
but still material for the proxy token, which is scoped to invoking
deployed endpoints — a leak means an attacker can run fits on the user's
account and see their results.

The reticulate design could promise that no token material entered R at
all. The HTTPS design cannot: tidylearn sets the auth headers itself, so
the proxy token is necessarily in the R process. The two credentials are
therefore treated differently.

**Mitigation.**

Account API token — unchanged, still never touched:

- tidylearn R code MUST NOT read `~/.modal.toml`, `~/.config/modal/`,
  `~/.modal/`, or any Modal token environment variable. The account token
  is used only by Modal's Python CLI at deploy time, outside R.
- No `Sys.getenv("MODAL_TOKEN*")` calls in tidylearn source.

Proxy token — held, but contained:

- Sourced from a single documented environment variable pair read at
  request time. tidylearn MUST NOT persist it, cache it in an R object
  that outlives the call, or write it into a fitted model.
- Attached with `httr2::req_headers_redacted()` so the values are masked
  in `req_dry_run()`, `last_request()`, printed request objects, and
  `httr2` verbosity. Plain `req_headers()` MUST NOT be used for the
  `Modal-Key` / `Modal-Secret` headers.
- Never interpolated into `message()`, `warning()`, `stop()`, or
  condition metadata.
- Accepting a token as a function argument is permitted only if it is
  consumed within the call and not stored; the environment variable is
  the documented path.

**Verification.**

- `grep -rn 'modal.toml\|MODAL_TOKEN\|MODAL_AUTH' R/ tests/` returns nothing.
- `grep -rn 'ak-\|wk-\|ws-' R/` returns no token-shaped literals (tests use
  opaque mocks).
- `grep -rn 'req_headers(' R/` shows no Modal auth headers set through the
  non-redacting variant.
- A test asserts that a constructed request's printed form does not
  contain the mock token value.

### T2: Silent data upload without user knowledge

**Scenario.** A user fits a model with `compute = "cloud"` and the
training data is silently uploaded to a third-party service without their
explicit awareness.

**Risk.** High. Even with a valid Modal account, the user may have
data-handling obligations (regulatory, contractual, or organisational)
that forbid silent egress to third parties.

**Mitigation.**

Two-layer consent (see [Section 4](#4-data-egress-consent-ux)):

- **Per-call:** `tl_model(..., compute = "cloud", confirm_upload = TRUE)`
  must be passed for every cloud submission, OR a session-level lock
  must be in place.
- **Session-level:** `tl_cloud_consent()` opts in for the rest of the R
  session and is documented as intended for batch / non-interactive use.
  The session lock is *not* persisted across R restarts.
- Before any data crosses the network, tidylearn prints a one-line
  summary of what is about to be uploaded (rows × cols, estimated MB,
  destination region) so the user can confirm the shape of the egress.
- A dry-run mode (`dry_run = TRUE`) prints the summary and exits without
  uploading.

**Verification.**

- A test that calls `tl_model(..., compute = "cloud")` without consent
  expects an error.
- A test that calls with `confirm_upload = TRUE` and a mocked submission
  layer verifies the upload-summary message fires before the submission
  call.
- `grep -rn 'cloud_submit\|req_perform' R/` flags every call site that
  performs a cloud request, so the consent gate is auditable.

### T3: Data persisted on Modal beyond job lifetime

**Scenario.** Training data lingers on Modal volumes or in Modal's
storage after the fit completes, beyond what the user expects.

**Risk.** Medium. Sensitive data accumulates in a third-party system
without the user actively choosing to store it there.

**Mitigation.**

- The Modal app definition tidylearn ships MUST NOT mount any Modal
  Volume by default. Job containers are ephemeral and destroyed after
  the fit returns.
- Persistent volumes are an explicit opt-in (a separate API, e.g.
  `tl_cloud_cache(...)`, not yet designed) for users who want to cache
  large datasets across fits. The opt-in path is out of scope for the
  first cloud-integration release.
- Returned model artifacts are streamed back to the R session and not
  persisted on Modal.

**Verification.**

- The Modal app definition file in tidylearn is reviewed for any
  `modal.Volume`, `modal.NetworkFileSystem`, or persistent-storage
  references.
- A test runs a cloud fit and then queries the Modal app's storage to
  confirm nothing was persisted (this is a manual integration test, not
  part of CRAN's automated suite).

### T4: Network interception in transit

**Scenario.** An attacker on the network path between the user and
Modal intercepts the upload, captures training data, or tampers with the
returned model.

**Risk.** Low (Modal endpoints are HTTPS-only) but worth naming. Higher
than under the reticulate design, because tidylearn now owns the transport
rather than delegating it.

**Mitigation.**

- All cloud transit uses `httr2` over HTTPS to Modal-hosted endpoints.
- tidylearn MUST NOT disable or relax TLS verification. No
  `req_options(ssl_verifypeer = 0)`, no `ssl_verifyhost` downgrade, no
  custom CA bundle.
- Any endpoint URL is rejected unless its scheme is `https` and its host
  is a Modal host (see [T9](#t9-egress-to-a-non-modal-host)).
- tidylearn MUST NOT call out to non-Modal endpoints during cloud
  submission (no telemetry, no analytics, no third-party storage).

**Verification.**

- The earlier form of this check — "no `httr*` in cloud paths" — no longer
  applies; `httr2` *is* the transport. The check is now about destination
  and TLS posture, not about the presence of an HTTP client.
- `grep -rn 'ssl_verify\|req_options' R/` shows no TLS downgrades.
- `grep -rn 'request(' R/` — every request base URL traces back to the
  validated endpoint from [T9](#t9-egress-to-a-non-modal-host), not to a
  literal or an unchecked user string.
- `grep -rnE 'curl::|download\.file|\burl\(' R/` returns nothing in cloud
  paths; `httr2` is the only client. (The `\b` matters — without it the
  pattern also matches identifiers ending in `url`, such as
  `tl_validate_modal_url(`.)

### T5: Telemetry / phone-home from tidylearn

**Scenario.** tidylearn collects usage data (which methods are run,
dataset sizes, model performance) and transmits it to a third party for
analytics, error reporting, or otherwise.

**Risk.** Medium. Even anonymised telemetry leaks information about the
user's work patterns and may be unacceptable in regulated environments.

**Mitigation.**

- tidylearn ships with NO telemetry. Period. No analytics, no
  crash reporting to a third party, no usage pings, no install-time
  callbacks.
- This is a project-wide policy, not specific to cloud compute, but it
  is restated here because the cloud path is the obvious vector for
  such a regression.

**Verification.**

- `grep -rn 'analytics\|telemetry\|track\|pingback\|sentry\|posthog'
  R/ tests/` returns nothing.
- CRAN review process is part of the defence (CRAN policy forbids
  phoning home without explicit opt-in).

### T6: Sensitive data in logs and messages

**Scenario.** tidylearn or the Modal SDK logs row-level training data,
model coefficients that encode sensitive values, or token-shaped
strings.

**Risk.** Medium. Logs may end up in CI artefacts, terminals, or shared
contexts.

**Mitigation.**

- tidylearn's `message()` / `warning()` / `cat()` output MUST contain
  only metadata: row counts, column counts, method names, timestamps,
  tier labels, cost estimates. Never row values, never coefficient
  values from sensitive fits, never tokens.
- The upload-summary message in [T2](#t2-silent-data-upload-without-user-knowledge)
  is constructed from dimensions, not data.
- Modal's own logs are user-readable and ephemeral; outside tidylearn's
  direct control but documented as such.

**Verification.**

- Code review: any new `cat()` / `message()` / `print()` in cloud paths
  is reviewed for data leakage.
- The existing `tl_compute_advisor()` print method is already
  metadata-only and is the template for cloud-path messages.

### T7: Compromised dependency exfiltrates data

**Scenario.** A dependency in the cloud path is compromised and uses that
path to exfiltrate user data. Under the HTTPS + R-worker design the
dependency surface is `httr2` on the client side, and the R packages
baked into the worker image on the remote side.

**Risk.** Low-to-medium. Hard to fully defend against, but the impact
surface can be limited. Smaller than under the reticulate design, which
put a Python interpreter and the Modal SDK's transitive tree inside the
user's session.

**Mitigation.**

- The worker image pins its **R** dependencies — a fixed `rocker/r-ver`
  base tag plus a package snapshot (a repository snapshot date or a
  committed `renv.lock`). Reproducible image builds, lock file committed
  to the repo.
- Because `method = "deep"` ships in the first cloud release, the worker
  image also carries a Python runtime with TensorFlow and keras, and
  **those are pinned too**. The R `keras` package and the Python keras it
  binds to must be a matched pair; an unpinned image would drift into the
  Keras 2 / Keras 3 split and fail at deserialisation rather than at
  build time. This is the cost of shipping `deep`: the other twelve
  methods need only R and CRAN packages.
- R-side cloud deps live in `Suggests:` so the install footprint is
  minimal. `httr2` is the only addition for the twelve pure-R methods.
  The `deep` path additionally calls `keras`, which tidylearn already
  suggests for local deep fits — it adds no new dependency, but it does
  mean reticulate and a Python runtime are present on the client for
  that one method.
- Direct dep tree is reviewed before each tidylearn release.
- This is an industry-wide problem; tidylearn does not solve it
  unilaterally but commits to good hygiene.

**Verification.**

- The worker image's pinned R dependency manifest is part of the
  cloud-integration PR and is reviewed, as is the pinned Python/TF
  manifest for the `deep` variant.
- The image build is reproducible from the committed manifest.
- `grep -rn 'reticulate::\|library(reticulate)\|require(reticulate)' R/`
  returns nothing — tidylearn never drives reticulate directly, even on
  the `deep` path, where it calls `keras::serialize_model()` and lets
  keras own the Python boundary. (Match on calls, not on the word: the
  source explains in comments why `python.builtin.object` is the right
  class to detect.)
- Cloud paths import only `httr2` and, for `deep`, `keras`.

### T8: Multi-tenant data leakage on Modal

**Scenario.** Another Modal user observes or accesses tidylearn user's
job data due to a Modal-side tenant-isolation failure.

**Risk.** Out of tidylearn's direct control.

**Mitigation.**

- Each tidylearn user deploys the Modal app to their own Modal account.
  tidylearn does NOT operate a shared, multi-tenant Modal service on
  behalf of users.
- Tenant isolation between Modal users is Modal's responsibility (see
  Modal's security docs).
- tidylearn's cloud documentation makes this explicit: cloud compute
  means the user's data on the user's Modal account, billed to the
  user, isolated from other Modal users by Modal's own controls.

**Verification.**

- The tidylearn Modal-deploy helper (`tl_cloud_setup()`) operates
  against the locally-configured Modal account. No tidylearn-operated
  endpoint.

### T9: Egress to a non-Modal host

**Scenario.** Because each user deploys their own Modal app, the endpoint
URL is user-supplied configuration. A typo, a copied-and-edited example, a
poisoned option in a shared `.Rprofile`, or a modified environment variable
sends training data to a host that is not Modal — while every consent gate
in [T2](#t2-silent-data-upload-without-user-knowledge) is satisfied and the
upload summary still reads as expected.

**Risk.** Medium-to-high. This threat did not exist under the reticulate
design, where the SDK resolved the destination from the user's Modal
account and tidylearn never named a host. Owning the transport means owning
the destination.

**Mitigation.**

- The endpoint URL is validated before any request carrying user data.
  It MUST parse, use scheme `https`, and have a host on the allowlist.
  Anything else is a hard error, not a warning.
- The allowlist defaults to Modal's own domains (`*.modal.run`, and
  `modal.com` for API hosts). Modal customers serving Web Functions from
  a custom domain extend it with `tl_cloud_allow_host()`.
- Extension is a **per-session function call**, never an option or an
  environment variable. A shared `.Rprofile` or an inherited environment
  must not be able to add an upload destination without the user having
  written the call. Additions are not persisted and die with the session.
- Added hosts are validated as bare host names. URLs, ports, paths and
  wildcards are refused, as is any single-label name — `"com"` would
  otherwise open an entire TLD and defeat this control completely.
- Host matching is anchored on a leading dot, for default and added
  hosts alike, so a host is matched only by itself and its subdomains.
- The pre-upload summary distinguishes a Modal host from a host added
  this session, so an extended allowlist is visible at the moment
  consent is acted on.
- Validation happens at the single choke point that builds the request,
  so no call site can bypass it.
- The pre-upload summary in [Section 4](#4-data-egress-consent-ux)
  displays the resolved destination host, so a wrong destination is
  visible to the user before they consent.
- Redirects are not followed for requests carrying user data — a 3xx to
  an off-Modal host would otherwise defeat host validation. (Modal's own
  150-second 303 continuation is a different mechanism and is handled by
  the documented poll loop, which re-validates each URL it is given.)

**Verification.**

- A test supplies `http://` , a non-Modal host, and a Modal-lookalike host
  (e.g. `modal.run.evil.test`) and expects an error in each case before any
  request is made.
- Tests confirm that adding a host does not widen anything above it: with
  `fits.example.com` allowed, `example.com`, `evil-fits.example.com` and
  `fits.example.com.evil.test` all remain refused.
- Tests confirm `https` is still required on an added host.
- `grep -rn 'request(' R/` confirms every request is constructed through
  the validating helper.
- Code review confirms redirect-following is disabled on data-carrying
  requests.

### T10: Runaway spend and orphaned jobs

**Scenario.** A user submits a cloud fit and is billed far more than they
expected. The ways this happens are mostly not adversarial:

- The R session goes away — Ctrl-C, a closed IDE, a crash, a closed
  laptop — and the job keeps running on Modal, because the session was
  only polling for a result and was never what kept the work alive.
- The job hangs or diverges and runs to whatever timeout it inherited.
- A retry policy silently multiplies the bill: Modal's timeouts are per
  execution attempt, so `n` retries can cost `n + 1` full timeouts.
- A submission is retried over a flaky network and runs twice.
- The advisor's estimate is optimistic and the real fit costs more.

**Risk.** High, and the consequence is financial rather than
informational. It was previously listed as out of scope on the grounds
that quota exhaustion is a usability concern; that was wrong. A library
that spends a user's money on their behalf owns this.

**Mitigation.**

The controls split into two groups, and the distinction matters more
than any individual control:

*Survives the client dying* — the only real guarantees:

- Every submission sets an explicit timeout, derived from the advisor's
  estimate with headroom and capped well below Modal's 24-hour maximum.
  tidylearn MUST NOT inherit Modal's default timeout, and MUST NOT
  submit a job with no timeout.
- Retries are off by default on the tidylearn worker function. A failed
  fit is the user's decision to repeat, not a silent cost multiplier.
- `tl_cloud_setup()` directs the user to set a spend budget on their
  Modal workspace. That budget is the only true hard cap, and it is not
  tidylearn's to set.

*Best-effort, lost if the session is killed:*

- A fit whose own estimate exceeds the timeout cap is refused before
  submission. Such a job would bill the full timeout and then be killed
  before producing anything.
- The user is shown, and gated on, the **worst-case** cost — timeout
  multiplied by the tier rate — not the expected cost. The estimate is
  order-of-magnitude; the timeout is the actual bound.
- A worst-case cost above `max_cost` is refused. Raising it is explicit.
- The submitted call id is recorded and printed, and in-flight jobs are
  listable, so a job is never invisible.
- Interrupt and error handlers cancel the remote call rather than
  orphaning it, and an explicit cancel is available.
- The submit POST is never automatically retried; only the idempotent
  result poll is.

**Verification.**

- A test asserts that a fit estimated beyond the timeout cap is refused
  rather than submitted.
- A test asserts the gate uses worst case, not the estimate, and that
  the refusal message says so.
- Tests assert the derived timeout has headroom over the estimate, a
  floor, and a cap.
- `grep -rn 'req_retry' R/` shows retry configured only on polling
  requests, never on submission.
- The Modal app definition sets an explicit timeout and `retries = 0`.
- Code review confirms in-flight jobs are registered before the request
  is performed, not after, so a response that never arrives still leaves
  a record.

## 4. Data egress consent UX

The egress contract has two layers:

**Per-call argument.** `tl_model(..., compute = "cloud", confirm_upload =
TRUE)`. Required on each call unless a session lock is in place. Default
is `confirm_upload = FALSE`, so naive calls error rather than silently
upload.

**Session lock.** `tl_cloud_consent()` opts in for the rest of the R
session. Intended for batch / non-interactive use where per-call
arguments would be repetitive. Not persisted across R restarts. Can be
revoked mid-session with `tl_cloud_consent(FALSE)`.

**Pre-upload summary.** Before any data crosses the network, tidylearn
prints:

```text
Uploading to Modal:
  Method:           xgboost
  Destination:      my-workspace--tidylearn-fit.modal.run
  Rows x cols:      1,500,000 x 47
  Estimated MB:     564
  Modal tier:       A10G (24 GB VRAM / 24 GB RAM)
  Estimated:        14 min, $0.43
  Timeout:          42 min (job is killed at this point)
  Most it can bill: $0.77
```

The destination line is part of the mitigation for
[T9](#t9-egress-to-a-non-modal-host): the host the data will actually reach
is shown before consent is acted on, and is marked when it is a host the
user added this session rather than one of Modal's own.

The last two lines are the mitigation for
[T10](#t10-runaway-spend-and-orphaned-jobs). The estimate is
order-of-magnitude and could be wrong; the timeout is what actually
bounds the bill, so the figure the user is asked to accept is the worst
case rather than the expectation.

The summary is generated from metadata only — no row values.

**Dry-run.** `tl_model(..., compute = "cloud", dry_run = TRUE)` prints the
summary and returns without submitting.

**Non-interactive contexts.** Rscript, CI, knit, etc. have no terminal
to prompt. In those contexts the only valid path is the session lock
(`tl_cloud_consent()` called earlier in the script) or per-call
`confirm_upload = TRUE`. tidylearn does NOT use `readline()` or any
interactive prompt that would break batch use.

## 5. Audit checklist (Modal-integration PR)

Before the Modal-integration PR is merged, reviewers must confirm:

- [ ] No `modal.toml` / `MODAL_TOKEN` / `MODAL_AUTH*` references in `R/`
      or `tests/`.
- [ ] No `Sys.getenv("MODAL_TOKEN*")` calls in `R/` (the account token is
      never read by R).
- [ ] No token-shaped string literals (`ak-...`, `wk-...`, `ws-...`)
      anywhere in source or tests (use opaque mocks).
- [ ] Proxy-token headers are set with `req_headers_redacted()`, never
      plain `req_headers()`, and a test asserts the token does not appear
      in a printed request.
- [ ] The proxy token is not persisted, cached beyond the call, or stored
      in a fitted model object.
- [ ] Every cloud submission call site is gated on either
      `confirm_upload = TRUE` or a session lock check.
- [ ] The pre-upload summary message is generated from metadata, not
      from row values, and names the resolved destination host.
- [ ] The Modal app definition includes no `modal.Volume`,
      `modal.NetworkFileSystem`, or persistent-storage references.
- [ ] No telemetry / analytics / pingback calls in cloud paths.
- [ ] The worker image's R dependency manifest is pinned and committed,
      and the build is reproducible from it. For the `deep` variant, the
      Python/TensorFlow/keras stack is pinned as a matched pair with the
      R `keras` package.
- [ ] Every request is built through the URL-validating helper; scheme is
      `https` and host is on the allowlist, enforced as an error.
- [ ] The allowlist is extensible only by an explicit per-session call,
      not by an option or environment variable, and added hosts are
      validated as bare host names with no single-label entries.
- [ ] Redirect-following is disabled on requests carrying user data.
- [ ] No TLS downgrades (`ssl_verifypeer`, `ssl_verifyhost`, custom CA).
- [ ] Every submission sets an explicit timeout; none inherits Modal's
      default or omits one.
- [ ] The Modal app definition sets `retries = 0`, so a hung job cannot
      bill several full timeouts.
- [ ] The submit request is never auto-retried; `req_retry()` appears
      only on result polling.
- [ ] The pre-upload summary states the worst-case cost and the timeout,
      and the budget gate refuses on worst case rather than estimate.
- [ ] A fit whose estimate exceeds the timeout cap is refused before
      submission.
- [ ] In-flight jobs are registered before the request is performed and
      are listable with `tl_cloud_jobs()`.
- [ ] `httr2` is the only HTTP client in cloud paths; no `curl::`,
      `download.file()`, or `url()`.

## 6. Out of scope

The following are intentionally not addressed by this threat model:

- Modal's internal infrastructure security (covered by Modal's own
  security docs and SOC 2 report).
- Local OS / file system security on the user's machine.
- R session-level isolation (e.g., another local user reading R objects
  from `/tmp` or memory).
- Threats requiring physical access to the user's machine.
- Cryptanalysis of TLS.
- Modal's own billing accuracy, and the spend budget the user sets on
  their Modal workspace. tidylearn bounds what it submits (see
  [T10](#t10-runaway-spend-and-orphaned-jobs)) but cannot enforce a cap
  on the account itself.

## 7. Open questions

These are commitments deferred to the Modal-integration PR:

- Exact API shape of `tl_cloud_consent()` (singleton vs. per-method
  granularity).
- How `dry_run = TRUE` interacts with the advisor — should `dry_run`
  also short-circuit the network-time portion of the cloud estimate?
- Behaviour when a user revokes consent mid-session while a fit is
  in-flight (likely: complete the in-flight fit, refuse new ones).
- Poll-loop behaviour on `404` from the result endpoint: Modal expires
  results after 7 days, which is indistinguishable at the HTTP level from
  a call ID that never existed.
- Logging cadence: at what verbosity level (`verbose = 0/1/2`) does
  the upload-summary print, and at what level (if any) is it
  suppressed?

Two questions listed here previously are now settled and described above:
the endpoint is read from the `TIDYLEARN_MODAL_ENDPOINT` environment
variable rather than an R option, and the host allowlist is extensible
via `tl_cloud_allow_host()` under the constraints in
[T9](#t9-egress-to-a-non-modal-host).

## 8. Revision history

- **2026-08-06** — added [T10](#t10-runaway-spend-and-orphaned-jobs),
  runaway spend and orphaned jobs, and removed the out-of-scope line that
  dismissed quota exhaustion as a usability concern. A submitted Modal
  job runs to completion regardless of what the R session does, so a
  closed session leaves a job billing invisibly; the mandatory
  submission timeout, not any client-side handler, is what bounds that.
  Section 4's summary now shows the timeout and the worst-case cost, and
  the gate refuses on worst case rather than on the estimate.
- **2026-08-06** — revised for the HTTPS + R-worker architecture. Modal has
  no R SDK and no general REST API for invoking Functions, so tidylearn
  calls a user-deployed Web Function over HTTPS with `httr2` instead of
  driving the Python SDK through reticulate, and the worker container runs
  R rather than Python. Consequences: T1 rewritten (tidylearn now handles a
  proxy token, so the "no token material in R" commitment is replaced by
  containment rules); T4 rewritten (the ban on `httr*` in cloud paths
  became a destination-and-TLS check); T7 now pins the worker's R
  dependencies rather than Python ones; T9 added for egress to a non-Modal
  host, a threat introduced by owning the transport. Sections 1, 4 and 5
  updated to match.
- **2026-05-22** — initial draft. Written alongside slice 4 of the
  compute-backends initiative (post local-GPU routing + advisor reframe;
  pre Modal-integration code). All mitigations described as commitments
  to be verified when Modal code lands.
