# tidylearn cloud compute — threat model

**Status:** DRAFT — describes the commitments tidylearn's cloud-compute
integration will meet when the Modal-backed `compute = "cloud"` path lands.
**Last updated:** 2026-05-22
**Scope:** the cloud-compute paths in tidylearn (`tl_compute_advisor()`'s
cloud-tier estimates, `tl_resolve_compute()`'s `"cloud"` branch, and the
forthcoming `tl_model(..., compute = "cloud")` submission to Modal).

**API names below are provisional.** `confirm_upload`, `dry_run` and
`tl_cloud_consent()` describe the intended design and do not exist yet;
`tl_model()` is the existing entry point they would attach to. The
commitments are binding, the spellings are not.

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
+---------------------+          +-------------------+         +-----------------+
| User's R session    |          | User's Modal      |         | Modal           |
| (trusted)           |          | account + token   |         | infrastructure  |
|                     |          | (user-controlled, |         | (third-party)   |
|  +---------------+  |   HTTPS  |  in ~/.modal.toml)|  HTTPS  |                 |
|  | tidylearn R   |--+--------->+---+---+-----+-----+-------->+ ephemeral       |
|  | code          |  |          | Modal Python SDK  |         | container       |
|  | (trusted)     |  |          | (via reticulate)  |         | (per job)       |
|  +-------+-------+  |          +-------------------+         +-----------------+
|          |          |
|  +-------v-------+  |
|  | local data    |  |
|  | (sensitive,   |  |
|  |  user-owned)  |  |
|  +---------------+  |
+---------------------+
```

Boundaries:

- **R session ↔ tidylearn code** — trusted in both directions. tidylearn
  runs in the user's R process with the user's privileges.
- **tidylearn code ↔ Modal SDK** — tidylearn calls into the Modal Python
  client via reticulate. The Modal SDK is treated as a sub-component
  inside the trusted tidylearn boundary, but its dependency tree is
  pinned and reviewed (see [T7](#t7-compromised-dependency-exfiltrates-data)).
- **Modal SDK ↔ Modal infrastructure** — TLS-encrypted, Modal's
  responsibility.
- **User's local data → Modal** — sensitive data crossing a third-party
  boundary. This is the most consequential transition and is gated by
  explicit consent (see [Section 4](#4-data-egress-consent-ux)).
- **User's Modal token** — never crosses into tidylearn code at all. It
  is held by the Modal CLI / SDK in `~/.modal.toml` and is read by the
  SDK, never by tidylearn (see [T1](#t1-token-exfiltration-via-tidylearn-source)).

## 2. Assets

| Asset | Sensitivity | Storage | Lifetime |
| --- | --- | --- | --- |
| Modal auth token | High | `~/.modal.toml` (user OS) | Until revoked |
| Training data (rows × cols) | User-determined; may contain PII | In-memory in R, transmitted to Modal during fit | Ephemeral on Modal — destroyed with container |
| Fitted model artifacts | Medium — may memorize training data | Returned to R session; not persisted on Modal | Lives in user's R session |
| Hyperparameters / model spec | Low | In-memory in R, transmitted to Modal | Ephemeral |
| Job stdout / logs | Low — should contain no row-level data | Modal log retention (per Modal's policy) | Per Modal |

## 3. Threats and mitigations

### T1: Token exfiltration via tidylearn source

**Scenario.** tidylearn code (or a future contribution to it) reads
`~/.modal.toml`, logs the contents to a message, transmits them to a
remote endpoint, or otherwise causes the user's Modal token to leak
outside the Modal SDK.

**Risk.** High. A leaked token gives an attacker the ability to spin up
Modal jobs (incurring cost) and access any data the user has previously
uploaded.

**Mitigation.**

- tidylearn R code MUST NOT read `~/.modal.toml`, the Modal token
  environment variables, or any path under `~/.config/modal/` /
  `~/.modal/`. Token access is delegated entirely to the Modal SDK's
  own initialisation.
- tidylearn MUST NOT pass token values as function arguments through R
  code. Any cloud submission goes via the Modal SDK's authenticated
  client; no token material crosses the reticulate boundary explicitly.
- No `Sys.getenv("MODAL_*")` calls in tidylearn source.
- No printing or message()ing of any string that includes a token shape
  (Modal tokens look like `ak-<32 chars>-<32 chars>`).

**Verification.**

- `grep -rn 'modal.toml\|MODAL_TOKEN\|MODAL_AUTH' R/ tests/` returns nothing.
- `grep -rn 'ak-' R/` returns nothing (tokens never appear as literals).
- Modal-integration PR review must confirm token paths are untouched by
  R code.

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
- `grep -rn 'cloud_submit\|modal\$\|modal::' R/` flags every call site
  that touches Modal, so the consent gate is auditable.

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

**Risk.** Low (Modal endpoints are HTTPS-only) but worth naming.

**Mitigation.**

- All transit to Modal happens via the Modal SDK, which uses TLS to
  Modal endpoints. tidylearn does not implement its own transport.
- tidylearn MUST NOT call out to non-Modal endpoints during cloud
  submission (no telemetry, no analytics, no third-party storage).

**Verification.**

- The Modal SDK's TLS handling is Modal's responsibility — out of scope
  for this doc.
- `grep -rn 'curl\|httr\|httr2\|download\.file\|url(' R/` against the
  cloud-related files shows no non-Modal network calls in cloud paths.

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

**Scenario.** A dependency in the cloud-submission tree (reticulate,
modal-python, an indirect Python dep) is compromised and uses the cloud
submission path to exfiltrate user data.

**Risk.** Low-to-medium. Hard to fully defend against, but the impact
surface can be limited.

**Mitigation.**

- The Modal Python app definition that tidylearn ships pins its Python
  dependencies (Modal SDK version + any libraries used in the container).
  Lock files committed to the repo.
- R-side cloud deps live in `Suggests:` so the install footprint is
  minimal.
- Direct dep tree is reviewed before each tidylearn release.
- This is an industry-wide problem; tidylearn does not solve it
  unilaterally but commits to good hygiene.

**Verification.**

- The pinned Python dependency file is part of the cloud-integration PR
  and is reviewed.
- `R/` source for cloud paths imports only `reticulate` (and through it,
  the Modal SDK) — no other R-side cloud deps.

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
  Method:        xgboost
  Rows × cols:   1,500,000 × 47
  Estimated MB:  564
  Modal tier:    A10G (24 GB VRAM / 24 GB RAM)
  Estimated:     14 min, $0.43
```

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
- [ ] No `Sys.getenv("MODAL_*")` calls in `R/`.
- [ ] No token-shaped string literals (`ak-...`) anywhere in source or
      tests (use opaque mocks).
- [ ] Every cloud submission call site is gated on either
      `confirm_upload = TRUE` or a session lock check.
- [ ] The pre-upload summary message is generated from metadata, not
      from row values.
- [ ] The Modal app definition includes no `modal.Volume`,
      `modal.NetworkFileSystem`, or persistent-storage references.
- [ ] No telemetry / analytics / pingback calls in cloud paths.
- [ ] Python dependency tree for the Modal app is pinned and committed.
- [ ] Non-Modal network calls (`curl`, `httr*`, `download.file`,
      `url()`) are absent from cloud-related R source.

## 6. Out of scope

The following are intentionally not addressed by this threat model:

- Modal's internal infrastructure security (covered by Modal's own
  security docs and SOC 2 report).
- Local OS / file system security on the user's machine.
- R session-level isolation (e.g., another local user reading R objects
  from `/tmp` or memory).
- Threats requiring physical access to the user's machine.
- Cryptanalysis of TLS.
- Adversarial inputs designed to exhaust Modal quotas (denial of
  service against the user's own Modal account by tidylearn callers
  — this is a usability issue, not a security one for this scope).

## 7. Open questions

These are commitments deferred to the Modal-integration PR:

- Exact API shape of `tl_cloud_consent()` (singleton vs. per-method
  granularity).
- How `dry_run = TRUE` interacts with the advisor — should `dry_run`
  also short-circuit the network-time portion of the cloud estimate?
- Behaviour when a user revokes consent mid-session while a fit is
  in-flight (likely: complete the in-flight fit, refuse new ones).
- Logging cadence: at what verbosity level (`verbose = 0/1/2`) does
  the upload-summary print, and at what level (if any) is it
  suppressed?

## 8. Revision history

- **2026-05-22** — initial draft. Written alongside slice 4 of the
  compute-backends initiative (post local-GPU routing + advisor reframe;
  pre Modal-integration code). All mitigations described as commitments
  to be verified when Modal code lands.
