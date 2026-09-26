# GitHub Actions architecture

GitHub Actions owns pull-request checks, package publication, and automated model discovery and onboarding.
Pull-request lint and packaging use GitHub-hosted runners, while the compiler-heavy test job uses the organization-level
`ubuntu-runners` builder group. Package publication remains GitHub-hosted. Agent-driven model work uses the separate
`agent-runners` group with the `agents` label because it can exceed ordinary hosted-runner limits and needs the tracked
skills and CloudRift inference endpoint.

## Workflow overview

| Workflow | Trigger | Runner | Result |
| --- | --- | --- | --- |
| **Tests** | Pull request to `main` | GitHub-hosted + `ubuntu-runners` | Runs Ruff, the complete test suite, and a PyPI package dry run. |
| **Publish to PyPI** | Manual dispatch or published GitHub release | GitHub-hosted | Verifies the source and distribution, publishes to PyPI, and optionally creates the release. |
| **Verify or onboard model** | Nightly schedule or manual dispatch | `agent-runners` / `agents` | Qualifies one available exact model/GPU deployment and updates the rolling lifecycle PR. |
| **Discover model** | Nightly schedule or manual dispatch | `agent-runners` / `agents` | Refreshes recipe lifecycle tags and onboarding shells in one rolling PR without renting a VM. |
| **Review agent prompts** | Nightly schedule or manual dispatch | `agent-runners` / `agents` | Reads the last discovery and qualification run and corrects one agent prompt when its wording caused the failure. |

There is no generic experiment workflow or GitHub dispatch input for `emmy bench`. Requested experiment runs start
from a developer checkout through the tracked `.agents/skills/run-experiment` skill.

## Pull-request body

`PULL_REQUEST_TEMPLATE.md` fixes the shape of every PR: a title that describes the change functionally, an
abstract of one short plain-English paragraph, an optional single artifact backing that abstract — a small
table, a diagram, a few lines of output — a horizontal rule, and then the rest under headings chosen to fit
the change. `Abstract` is the only fixed heading. The split exists so a reader can decide from the abstract
alone whether the PR concerns them, which is why code references stay below the rule. The template also asks
for two revision passes before posting, because the failure it guards against is a first draft that lists
everything done instead of saying what the change is.

The template stays reusable and never holds one PR's content. Agents draft the body in an untracked temporary file
outside the repository, post that body to GitHub, and leave the tracked template unchanged.

## Pull-request checks

**Tests** runs four parallel jobs and installs the CI dependency set on Python 3.13. A newer commit cancels the
previous run for the same pull request. The GitHub-hosted lint job runs Ruff check and format verification. The
compiler-heavy job uses `ubuntu-runners` for `make test`, including `tests/github/` coverage for helpers under
`.github/scripts/` and `.github/workflows/scripts/`. Hugging Face downloads used by tests are cached because anonymous
shared-runner traffic is rate-limited. A separate GitHub-hosted bare-Python job runs `make pypi-dist`, the exact
non-publishing build path used by the release workflow, and requires one wheel and one source distribution. This
workflow has no write permission and does not use deployment credentials. The test step has a 38-minute execution cap
and reuses the environment installed before that step; the outer 45-minute job allowance also covers dependency
installation and cache setup.

The native-runtime job runs Rustfmt, Clippy with warnings denied, and locked Cargo tests on a GitHub-hosted runner.
These checks require no GPU. Native GPU parity and failure recovery run through `make test-native` on supplied hardware.

## Package publication

**Publish to PyPI** has two entry paths:

- Manual dispatch reads the version from `pyproject.toml`, refuses an already-tagged version, publishes to PyPI, then
  creates the matching tag and GitHub release.
- A manually published GitHub release must already have a tag matching `pyproject.toml`; the workflow validates and
  publishes that version without creating another release.

Both paths accept any commit already on `main`, including direct commits. Publication does not require a merged pull
request or a successful **Tests** workflow. Pull-request checks own lint, the complete suite, and the release-path
package dry run. `make pypi-dist` installs its minimal build dependencies, then uses `scripts/prepare_dist.py` to stage
bundled recipes and rewrite repository-relative README links for PyPI before building the wheel and source distribution.
The release gate requires exactly one of each,
installs the wheel into a clean environment, checks its version, and reads its bundled recipe catalog. The build
artifact moves between jobs through GitHub artifacts. PyPI uses trusted publishing through the `pypi` environment and
an OIDC token, so the repository stores no PyPI password. The manual path creates its GitHub release only after a
successful upload, preventing a failed publication from leaving a release behind.

## Model discovery and onboarding

All discovery paths use the tracked `discover-models` skill. The agent scores every existing recipe, selects exactly
ten complete recipes for the maintained set, makes conservative obsolete proposals, and proposes new onboarding
models. Repository code derives every remaining complete recipe as best-effort and restores every existing onboarding
shell from inventory with its task and deployment matrix unchanged. Every existing recipe and selected new model
receives a 0-100 heat score for current onboarding priority. Each promising new open-weight Hugging Face model becomes
an onboarding shell with one to three proposed deployment entries made only from `deploy.gpu` and
`deploy.gpu_count`; there is no shell-count limit.

An exact-SHA `emmy recipe query` reads the rolling `recipes/` root and expands its deployment rows. The workflow's
tracked `discovery_task.jq` filter groups those rows into recipe records and bounded scoring batches. The skill's
lifecycle and scoring prompts are attached from that same workflow commit, so the skill and GitHub Actions share one
prompt source. Three source investigators collect independent demand evidence, then hidden scorer subagents score the
deterministic batches without selecting lifecycle states. Hidden fit subagents size one new candidate each, in
parallel, reading that checkpoint's published configuration and the `emmy/gpu.py` capacity registry under the shared
`prompts/model-fit.md` contract; the parent relays their deployments and authors no hardware itself. The parent
returns only scores, maintained IDs, obsolete proposals, new onboarding models, and the sized deployments. The tracked
`discovery_manifest.jq` filter validates exact score coverage, ignores already-inventoried IDs repeated as new
candidates, and mechanically assembles the four-list manifest before the lifecycle validator applies policy. An
exact-SHA recipe query against the rolling root enforces the maintained count after application.

A rejected selection is not a failed run on its own: the step resumes the same OpenCode session with the exact
rejection and accepts a corrected selection, twice, before failing. A reply carrying no JSON object counts as a
rejection too, since it matches nothing in the filter, which would otherwise succeed having written an empty manifest
and fail the run two steps later on a parse error. The rejection names the offending IDs and the set to choose from,
because the agent assembles its answer from subagent reports with the batch rows long out of context; for the same
reason the task states the selectable set once as `maintainable_model_ids` rather than only as a per-row flag. The
step prints one line per agent event: a run in progress is visible only through the job log, and a rejected decision
has to stay readable afterwards.

The workflow checks that the agent did not modify the checkout, then validates and applies its lifecycle manifest. Its
artifact worktree remains on the rolling lifecycle branch, while the catalog, workflow scripts, OpenCode agent and
plugin directory, attached discovery skill, and prompt files come from the exact `github.sha` that started the run.
This lets a manual dispatch test a workflow PR without copying its implementation commits into the rolling branch or
silently using an older manifest contract. The manifest filter reads the last fenced or bare object carrying exactly
the five expected selection fields, so reasoning before or after it is tolerated, and requires exactly the five
expected selection fields before assembling the manifest. Only new candidates are sized: an existing onboarding
shell keeps the matrix it was created with, because sizing it again every run only reshuffled its hardware. An empty
sized result drops a new candidate that nothing in the fleet can serve. The named discovery agent denies repository
edits and permits only the tracked discovery skill, public-web tools, repository reads, read-only Git inspection, the
three named read-only source subagents, the tool-free batch scorer, and the fit subagent. Parent work caps at 64
agentic steps. The Reddit, Hugging Face, and OpenRouter/Arena investigators run as independent bounded sources; Reddit
can surface a candidate before an exact Hugging Face identity is known. The last complete selection object in
OpenCode's final completed text event is logged before deterministic assembly so a rejected decision remains
inspectable; the repository validator remains the authoritative completion gate. The project provider configuration
selects the configurable CloudRift model through an OpenAI-compatible Chat Completions endpoint and disables the
model's chat-template thinking mode for the concise JSON result. Discovery never provisions hardware.

OpenCode is provisioned on the self-hosted runners rather than maintained inside Emmy. `.opencode/opencode.json` owns
the model provider alias, while `.opencode/agents/` owns the separate discovery and onboarding limits and permissions.
Both live under `.opencode/` because the workflows point OpenCode's config directory at the exact workflow source,
which loads after the checked-out branch's config; a provider setting anywhere else would come from that branch. The
tracked `.agents/skills/` remain the canonical task definitions. Compatibility symlinks under `.claude/skills/`
expose the same packages through OpenCode's native skill tool.

### Nightly verification and direct onboarding

**Verify or onboard model** runs nightly and retains a manual exact model/GPU dispatch. Its selector uses the
`emmy recipe query` command against the rolling branch's recipe root, with the command implementation loaded from the
exact workflow SHA. Manual dispatch supplies one exact external candidate; scheduled dispatch queries declared
deployments. A filtered-out manual candidate is an error, while no scheduled match is a successful no-op.
The query's filters and sorts read CloudRift VM variant availability without filtering on public-IP supply and consider
only declared deployments with an available exact CloudRift GPU count. Pending `onboarding`/`untested` recipes are the
first priority, ordered by descending heat, then model ID and deployment declaration order. If none can run, the
selector performs a second generic query for a `maintained` recipe whose committed `RESULTS.md` has the oldest
last-change timestamp; a missing report is oldest. No eligible deployment is a successful no-op.

The workflow requires the repository's `CLOUDRIFT_TEAM_ID` variable to contain the exact Robots team UUID. Before it
checks capacity, it validates that `CLOUDRIFT_API_KEY` can act for that UUID through a team-scoped account request;
every rent then includes the UUID and requests a public IP so the GitHub runner can reach the VM over SSH. It attaches
`emmy`, workflow, and GitHub job tags,
makes at most three workflow-level rental attempts for the same selection, and sweeps a failed attempt by the complete
tag set before retrying. Only V100 rentals set CloudRift's admin-only billing exemption; every other GPU is a regular
team rental. The workflow never falls back to GCP or changes the selected GPU type/count.

Unconditional teardown sends the complete tag-scoped terminate request before its bounded status audit and lease
audit. This ordering gives cancellation cleanup a short critical path inside GitHub's cancellation grace period while
retaining the owned lease as an independent verification handle.

The workflow passes the resulting SSH target and an explicit `onboarding` or `verification` mode to the tracked
`onboard-model` skill. Onboarding replaces the discovery shell and changes `onboarding`/`untested` to `best-effort`.
Verification begins from the active recipe, refreshes measurements and durable artifacts, and preserves its existing
lifecycle tag. Both modes preserve discovery-managed `model.heat`. Before the agent starts, the workflow installs the
small remote Python/rsync prerequisite set and
requires `$HOME/.cache/emmy` to be durable storage with at least 8 GiB free. Compiler staging keeps its checkout,
venv, cache, and build temporary files there rather than on a small `/tmp` tmpfs. The job has a 24-hour limit and gives
the agent a 23.5-hour deadline so artifact validation and cleanup retain 30 minutes. The deadline is the agent's only
budget. It has no step cap: a capped agent can only answer in text once it reaches the cap, so it never writes its
summary. An agent that ends without a summary fails its step, and the failure notice says so. For the selected
recipe and GPU, the same nightly qualification validates the recipe-local golden schema, strictly decodes every stored
row, and replays it on the exact card; pull-request tests do not load checked-in golden files. The shared serving
experiment retains one LFS archive per exact GPU platform plus one cumulative `RESULTS.md`; each archive includes its
system-only row records, and a run replaces only its platform snapshot. Ignored dated run directories, loose benchmark
output, top-level row-record copies, and qualification summaries are not repository artifacts. An Emmy-tuned prebuilt
image is produced only when every release gate passes. Nightly image publication is disabled unless
`NIGHTLY_ONBOARD_PUBLISH_IMAGE` is `true`; manual dispatch retains an explicit input.

The artifact worktree stays on the rolling lifecycle branch, while Python control code is loaded from the exact
`github.sha` whose workflow definition started the job. This keeps normal scheduled runs reproducible and lets a
manual dispatch test a workflow PR without leaking that PR's implementation commits into the model-artifact branch.
The selector runs the exact-SHA catalog logic against the rolling worktree's `recipes/` directory so lifecycle mode
and priority always reflect the branch that the agent will update.
The workflow also attaches the exact-SHA README related-project map, the `onboard-model`, `tune-kernels`, and
`run-experiment` skills, and the `prompts/onboard-model/` qualification, benchmarking, and investigation prompts as
authoritative agent inputs. It loads the OpenCode agent and plugin directory from that same commit; older copies on the
rolling branch cannot silently override a proposed artifact contract. As in discovery, the workflow renders only a
compact task object — mode, model, exact target, SSH handles, deadline, publication authorization, and summary
path — while the shared prompts own the run policy, so the skill and GitHub Actions share the same prompt sources.
The named onboarding agent may delegate bounded read-only compatibility research or failure diagnosis to the hidden
`onboard-investigator` subagent, which follows the attached investigation prompt. The parent retains every edit and
measurement. When an official engine image tag is missing, qualification checks official registries, release notes,
and upstream documentation for a renamed repository or current compatible tag before failing, then pins the exact
working tag or digest. A necessary small, model-independent compatibility fix may touch at most eight
implementation/test/architecture files and 500 changed lines, and any Python source change requires a focused test
change. Broader changes fail artifact validation and remain follow-up work.

The agent returns an atomic manifest. `.github/scripts/onboarding_artifacts.py` accepts only declared changes under the
allowed recipe (including `golden/<gpu-slug>_<compute-cap>.yaml`), experiment, serving-image, and bounded
implementation/test paths. The validator requires the shared experiment recipe and report plus the exact
`results_<gpu-short>x<gpu-count>.tar.gz` archive, and it opens that archive to require matching current-platform row
records.
It rejects changes to another platform snapshot and requires the current archive to be created or updated. Optional
outputs must remain in `artifacts`; unmanifested or exploratory output is rejected. The job installs a pinned,
checksum-verified Git LFS binary in the runner's temporary directory when needed, then configures LFS locally before
staging so the normal push uploads the archive object with the rolling branch. After the agent returns, the workflow
requires each task-local timestamp directory to be a root member of the declared platform archive before removing
that ignored local directory. It deletes current-platform top-level record copies only after the archive passes its
record and byte-read checks; only the durable archive proceeds to staging.
The workflow writes the agent's final completed text event to the job log before checking its exit status, preserving
the failure explanation after temporary output cleanup. The validator also checks the requested mode, exact recipe
model, expected lifecycle tag, and compact deployment and measured-performance summaries from the selected recipe
lane. The workflow then commits those artifacts, rebases on the latest default branch, and updates or opens the
rolling model lifecycle PR using renewable GitHub App credentials for the long-running push path.

Both lifecycle workflows finish with a separate GitHub-hosted notification job. Discovery groups only recipe entries
actually modified by the run under their resulting lifecycle, includes each current heat score, and links the run and
rolling PR. Onboarding includes the selected model, target, operation mode, serving deployment, and measured
performance from its validated atomic summary. A failed summary marks a regression only when a previously qualified
behavior or measured lane cannot be restored by a bounded fix; the isolated notification job then sends a prominent
red Discord notice with the credential-free gate and message. Because the notification job is independent of the
self-hosted agent job, it still runs after a failure, cancellation, or timeout. Discord delivery retries three times,
remains non-blocking, and disables all mentions; the workflow run, durable reports, and rolling PR retain the complete
evidence.

### Nightly prompt review

The discovery and qualification agents are driven entirely by the Markdown under `prompts/` and `.agents/skills/`, so
a failure that traces to a sentence there repeats every night until someone reads a log. **Review agent prompts** runs
at 05:00 UTC, before both, and closes that loop: it takes the most recent completed run of each, and stops without
starting an agent unless one failed or emitted a warning annotation. Annotations rather than whole logs are what make
a quiet night nearly free, and they are also what catches a run that eventually succeeded after burning its correction
budget.

The review is deliberately hard to use. `prompts/review-agent-prompts.md` states a four-part bar — a real failure, a
cause traceable to the prompt as written, a repeat on the next run, and one sentence that would prevent it — and names
the cases that are never prompt defects: infrastructure faults, correct negative conclusions, one-off flakes, and
anything whose real fix is code or a permission rule. Changing nothing is the expected nightly outcome. The workflow
enforces the rest mechanically rather than trusting the verdict: the agent may only modify tracked files under
`prompts/` and `.agents/skills/`, at most two files and fifteen lines, and a verdict that disagrees with the working
tree in either direction fails the run. Loosening a boundary, authorization, or safety rule is forbidden outright,
because an agent blocked by such a rule is usually the rule working.

A correction lands as one commit on the rolling discovery branch with a comment on the PR, so it reaches the nightly
agents only once a person merges it. That is the review's real safety property: it proposes, and a human still
decides. It shares the `model-discovery` concurrency group, and `.github/workflows/scripts/rolling_pr.sh` holds the
one copy of the rolling-branch lookup and force-with-lease rebase that it and **Discover model** both use.

### Discovery lifecycle PR

**Discover model** runs nightly or by manual dispatch. Discovery and qualification share one rolling draft PR rather
than opening one PR per model, but each holds only its own concurrency group: a qualification run keeps a rented GPU
for up to a day, and serialising the two behind one group made every discovery run wait for it. What they share is the
branch, so each does its long work on its own checkout and replays its commit onto the rolling branch as it stands at
push time, retrying when the branch moved underneath. A conflict there is a genuine overlap and fails the run. Each
workflow fails closed if more than one rolling PR exists. It also adopts one unpaired
discovery branch left by an interrupted PR-creation step, while
failing closed if multiple such branches would make ownership ambiguous. Before rendering inventory or running the
agent, it rebases an existing rolling branch onto the latest default branch. The rebase push uses the exact original
remote head as its force-with-lease expectation; a conflict, a stale checkout, or a concurrent branch update stops the
run before any lifecycle changes are applied.

The validated manifest tags the ten selected complete recipes `maintained`, keeps other useful recipes runnable as
`best-effort`, and uses `obsolete` only when the rationale names the exact ID of an all-around better maintained or
best-effort replacement for the same task at a comparable or lower practical VRAM footprint, or gives a technical
reason the recipe should no longer be used. The manifest must classify and score every complete recipe exactly once.
For decisions with a replacement, the validator demotes the proposal to `best-effort` unless the replacement is active
and serves the same task. A replacement described as merely comparable, or whose recipe reduces configured context or
concurrency, also defaults to `best-effort` while retaining the supplied heat. No repository code estimates whether a
checkpoint fits a platform: memory footprint depends on total parameters, quantization, and context, which the recipe
does not record. `prompts/model-fit.md` is the shared contract where that reasoning happens, attached to both the
discovery and onboarding agents so a proposed platform and a measured one mean the same thing. Unknown or malformed
lower-priority model IDs cannot stand in for an omitted recipe because every real recipe must still be scored. A
checkpoint name is normalized across a missing or incorrect organization only when it uniquely identifies one existing
recipe; ambiguous or unknown maintained IDs still fail validation because all ten selections must resolve exactly. The
agent must use `best-effort` when the old model retains any material capability or operating advantage. Every complete
recipe stores its rationale and heat immediately after `model.huggingface`. Every run scores every recipe afresh,
rewording the rationale and moving the heat a few points even when nothing changed, so a recipe keeps its recorded
pair until its lifecycle changes or its heat moves by at least 10; rewriting both every run buried the real changes
of each rolling PR. Obsolete recipes remain in git but cannot be deployed, benchmarked, published, or bundled; a later
reassessment may return one to the maintained or best-effort set.

The workflow creates every selected `onboarding`/`untested` shell through the same catalog library that backs
`emmy recipe create`. Each shell stores its rationale and heat under `model` and a list of one to three candidate
deployment entries under `matrices`; subsequent runs preserve the task and setups mechanically and refresh heat and
rationale under the same rule. A shell does not claim qualification. The workflow commits lifecycle updates to the
rolling branch and uses the API-only `make setup-agent` target for repository helpers plus `gh` for rolling-PR
discovery and updates. It never rents a VM. Network operations use bounded retries, and discovery keeps source
evidence, batched recipe context, retained history, and final output within the inference endpoint's context limit.
The workflow filters perform only structural batching and manifest assembly; the lifecycle validator retains
classification policy and manifest application.

## Credentials, VM ownership, and cleanup

OpenCode inherits the CloudRift inference key only for provider requests. The project plugin removes CloudRift, GCP,
GitHub Actions, and GitHub CLI credentials from every agent shell subprocess. Onboarding retains only the explicitly
required Hugging Face and Docker Hub credentials. The self-hosted runner must not carry unrelated ambient cloud
credentials.

`emmy vm create gpu --lease` writes a run-owned lease as soon as CloudRift returns an instance ID. The lease binds the
provider handle, exact request, workflow owner, and SSH target. Cleanup first deletes and audits that handle, then
lists and terminates every still-active CloudRift VM carrying the complete run-unique tag set. The tag audit catches a
VM created before the lease was durable without selecting another job's rentals. An `if: always()` step performs both
paths after OpenCode exits and fails the job if either ownership audit leaves a VM active. A runner that dies
mid-job runs no further step, so each onboarding run first terminates every VM carrying the workflow's tags other
than the job tag: the concurrency group serializes runs, so such a VM belongs to a run that is already dead.

GitHub App credentials are used for long-lived branch writes and PR operations. Private keys and temporary provider
configuration live only under run-specific `/tmp/emmy-*` paths and are removed by unconditional cleanup steps.

## Repository configuration

Agent workflows use these repository secrets as applicable:

- `CLOUDRIFT_API_KEY` for model discovery, Robots-team resolution, availability, and CloudRift provisioning;
- `DISCORD_EMMY_ROBOTS_WEBHOOK_URL` for non-pinging model discovery, verification, and onboarding summaries;
- `HF_TOKEN` for gated checkpoints;
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for an eligible verified prebuilt image.

`ONBOARD_AGENT_MODEL` selects the discovery/onboarding model and defaults to `Qwen/Qwen3.8-27B-FP8`.
`CLOUDRIFT_TEAM_ID` must be the exact Robots team UUID; the verification/onboarding workflow fails before capacity
selection if the variable is absent, malformed, or inaccessible to `CLOUDRIFT_API_KEY`.
`CLOUDRIFT_INFERENCE_URL` selects its OpenAI-compatible endpoint and defaults to
`https://inference.cloudrift.ai/v1`.
`NIGHTLY_ONBOARD_PUBLISH_IMAGE=true` authorizes a nightly qualification to publish an otherwise eligible image; it is
false when unset.
