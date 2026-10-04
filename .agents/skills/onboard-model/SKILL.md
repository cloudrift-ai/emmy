---
name: onboard-model
description: >-
  Onboard or periodically reverify and benchmark a Hugging Face model on an exact target GPU platform. Use when asked
  to add a model recipe, refresh a maintained recipe on a supplied GPU server, benchmark serving, create reproducible
  experiments and a durable results report, fully qualify and record the model's Emmy compiler inventory even when
  serving is blocked, or publish a prebuilt CloudRiftAI serving image.
---

# Onboard or reverify a model

Turn a Hugging Face model ID, an operation mode, and an exact `(GPU name, GPU count)` into reviewed repository
artifacts:

- a complete, optimized compiler golden under `recipes/<model>/golden/`, when the card runs Emmy and coverage is
  complete;
- one recommended serving recipe under `recipes/<model>/recipe.yaml` — an Emmy recipe with a verified, prebuilt
  `cloudriftai/vllm-emmy-<model-slug>:<tag>` image when Emmy qualifies, otherwise a vLLM or SGLang recipe;
- one reusable serving experiment under `experiments/<model>/` with a cumulative report and one Git LFS archive per
  exact GPU platform;
- `recipes/<model>/RESULTS.md`: the report beside a valid recipe, or a dated failure record when no recipe qualifies.

The experiment root keeps `recipe.yaml`, one cumulative `RESULTS.md`, and `results_<gpu-short>x<gpu-count>.tar.gz`
per measured platform (`<gpu-short>` from `emmy.hardware.gpu_short_name`, e.g. `results_rtx4090x1.tar.gz`). The
archive holds the timestamped raw run with its system-only experiment records; never commit those records as
top-level files. Track archives with Git LFS. When the caller says LFS is configured, verify the archive's attribute
but do not modify or list `.gitattributes`. Never commit the ignored dated run directory, loose benchmark output,
plots, compiler run summaries, partial working goldens, or onboarding summaries.

All platforms that share a protocol use one experiment root. A platform run replaces only its own archive and its own
section of the shared `RESULTS.md`, and removes legacy top-level records only for its platform after verifying the
archive keeps them. Reuse an existing root that represents the protocol, e.g. `serving_v100_sxm2_16gb`; otherwise
create `experiments/<model>/serving/`.

Use only the supplied SSH server. The caller owns VM creation and deletion; this skill owns everything on the node and
must tear down every workload before returning. Never switch GPU type, count, provider, quantization, or checkpoint to
rescue a failed run. The node is rented for this run alone and its user has passwordless sudo, so finishing its
provisioning is part of the work: `emmy deploy ssh` installs Docker and adds the user to the docker group (reconnect
before retrying, because the group applies only to a new login). A missing package, stopped daemon, or missing group
is a node to provision, never a gate failure. Reach the host through Emmy's own commands, not bare-shell probes.

## 0. Inputs and mode

Read [`README.md`](../../../README.md) first. Its **Related Projects** map names the platform, runtime, provisioning,
and serving repositories that may own relevant compatibility behavior; consult every relevant one, and verify its
remote source when compatibility may have changed, before concluding that a model, engine, or GPU is unsupported.
They are approved research and source-build candidates, not places to write or publish.

An **interactive run** asks the developer for missing inputs and reports in conversation. An **automated run** reads
[`prompts/onboard-model/qualify.md`](../../../prompts/onboard-model/qualify.md) completely first; it defines the task
payload, the caller-owned boundaries, and the artifact contract, and it never asks questions. Inputs:

1. the mode, exactly `onboarding` or `verification`;
2. the exact Hugging Face model ID;
3. the exact `emmy/hardware.py` GPU name and the GPU count;
4. an SSH target accepted by `emmy deploy ssh` / `emmy bench --ssh`;
5. a wall-clock deadline;
6. whether image publication is authorized, with Docker Hub credentials when it is;
7. whether a multimodal model is qualified as `multimodal`, `text-only`, or `auto` (the checkpoint's advertised
   modality);
8. for an automated run, an absolute output path for the summary.

In an automated run, a missing or ambiguous input is an immediate failure, and the publication input is the approval:
publish only when it says so, without a conversational pause. When the caller supplies an environment file, source it
in the same shell invocation as each `emmy` command — `.env` first when it exists, then the named overlay; in CI use
the injected environment and never guess an overlay. Check that required variables are set without printing them.

Before changing files, verify the model exists on Hugging Face, the host reports exactly the requested GPU name and
count, and the checkout is on a feature branch. Model and GPU selection, VM rental and cleanup, git, and pull requests
belong to the caller.

- `onboarding` starts from the existing `onboarding`/`untested` shell (a direct caller may authorize a new recipe).
  Keep its `model.heat`, and replace the pending tags with `best-effort` only when a valid recipe qualifies.
- In both modes, a successful run removes an `onboarding-failed` tag, and a failed run adds it (section 6).
- `verification` starts from the existing active recipe, refreshes its measurements and artifacts, and keeps its
  lifecycle tag and `model.heat`. It never changes the checkpoint, GPU/count, lifecycle, or heat.

Both modes run every section below; for a recipe without an Emmy serving configuration, sections 2 and 4 are the
point of the run. Any other change to an existing recipe needs explicit caller authorization.

**Bounded fixes.** You may make a small, model-agnostic fix under `emmy/` when this exact qualification needs it: a
serving compatibility fix, a compiler coverage gap, or a lowering or schedule fix that closes a kernel loss. Prefer an
existing mechanism, add a focused test or a realization-corpus case, update the nearest `ARCHITECTURE.md`, and
validate the reproducer before resuming. No broad refactor, dependency upgrade, validation bypass, workflow change,
fusion gate, or model-specific hard-code. A fix that does not validate or cannot finish before the deadline is
reverted; record its diagnosis in the report instead.

## 1. Research the serving path

From the model card, `config.json`, and current engine documentation, record:

- architecture, modality, total and active parameters, dtype and quantization;
- the VRAM fit on the requested platform per [`prompts/model-fit.md`](../../../prompts/model-fit.md), with its
  arithmetic; when the weights cannot fit, the fit gate fails and nothing is deployed;
- the immutable Hugging Face commit for this run;
- native context length and any documented practical cap;
- current vLLM or SGLang support and the first pinned image that supports it;
- required tool-call, reasoning, tokenizer, and multimodal flags;
- known model-specific launch or correctness issues.

Prefer vLLM when both engines support the checkpoint. If an image cannot be pulled, check the official registry,
release notes, and upstream repository for a renamed repository or a current compatible tag; use a moving tag only
for diagnosis, then pin the exact tag or digest. Never use an unofficial image from outside the **Related Projects**
map without explicit authorization. Read `emmy/recipe/ARCHITECTURE.md` before authoring YAML; a named recipe field
must not be repeated in `extra_args`.

Up to three independent read-only questions — model protocol, engine or image compatibility, diagnosis of one captured
failure — may go to the onboarding investigator with the complete
[`prompts/onboard-model/investigate.md`](../../../prompts/onboard-model/investigate.md) and exactly one question
each. You own every command, edit, measurement, and conclusion, and investigation never eats the time reserved for
artifacts and cleanup.

## 2. Build and optimize the golden

When the requested card has a compute capability Emmy's CUDA backend accepts (`emmy/gpu.py`), this is the first step
after research, and it runs even when fit, engine support, or another serving gate has failed: the golden is a
durable deliverable on its own. A card with no CUDA compute capability skips this section and section 4; record Emmy
as ineligible for that reason. Spend at most half of the remaining deadline here, so serving still has time.

**Coverage.** Build a manifest from the immutable configuration that maps every layer index and non-layer seam to a
traced representative: embeddings, final normalization and output head; every attention type, sliding/global pattern,
compressor, or recurrent path; dense MLP, routed-expert, shared-expert, and MTP paths; layer-local gates, residual
layouts, and materially different storage or quantization layouts. Prefer one architecture-only whole-model trace.
When that is unbounded for a very large checkpoint, trace each distinct path and merge the programs into one working
golden, deduplicating only identical programs and target identities, never by kernel name. A representative expert is
valid only where Emmy deliberately keeps routing, sort, and combine on the host; never stub an unsupported GPU
operation to claim coverage. Coverage is **complete** when every manifest path emits a non-empty target set and every
target reconstructs and lowers for the exact compute capability; otherwise it is **partial**. Fix tractable gaps with
a bounded fix and retrace. The layer-to-program mapping goes in the summary; the report gets only counts and gaps.

**Recording.** Record every target with `emmy run --golden <working.json> --bench --record`, and the kernel set the
greedy pick took with `--record-greedy`, then repeat the deployable O3 correctness check on the recorded rows. Every
retained target needs a deployable O3 measurement, a positive reference-backend measurement, and a `torch.compile`
measurement. When no recorded row beats the greedy pick, record the correct greedy pick rather than dropping the
target.

**The bar: every target on par with `torch.compile` or faster.** Bench the working golden at deployable optimization:

```bash
emmy run --golden <working.json> --bench --bench-backends eager,tcompile,emmy --strict --json <out>
```

Read the `--json` record (`record_knobs`, `status`, `flags`, `lane` per row), never the terminal table; the clean rows
it benches land in the tune DB as measured evidence. Rank targets by `tcompile_us / emmy_us`, losers first, and
classify each loss:

- a **missing measurement** is fixed by measuring more — bench the schedule a sibling uses with `--ab` and record it;
- an **eligibility or optimization lockout**, a pin that refuses or fails to lower, or a pin that runs wrong gets a
  bounded fix when one is tractable, and otherwise becomes a realization-corpus case (below);
- a **code generation quality** loss is reported, not recorded;
- a schedule the compiler correctly refuses is not a gap.

Repeat until every target meets the bar or the time for this section is spent. Record each remaining loser with its
ratio and class in the report.

**Corpus cases**, at most five per run: read `tests/compiler/realization/ARCHITECTURE.md` first. Name the desired
schedule with cited evidence — a sibling card's golden carrying that family for the same structural identity, the same
family already winning at a neighbouring binding, or a roofline argument — and put it in the case's `note` as an
`evidence:` paragraph. Minimize to the smallest reproducing snippet:

```bash
case=tests/compiler/realization/cases/<family>/<name>_xfail_<stage>.json
emmy trace -c "<snippet>" --target sm_<cc> -o "$case"
# set the realization's "knobs" from record_knobs, and "note": "evidence: ..."
make test-corpus-regen   # restamps the case onto the fresh lowering; the note stays
```

The case must fail without its suffix and pass with it; record the command. A stale existing case is reported, not
regenerated — restamping belongs to the change that invalidated it.

**Commit.** With complete coverage, strip working `ranking` metadata and commit the self-contained document as
`recipes/<model>/golden/<gpu-slug>_<compute-cap>.json`, one file per exact GPU and compute capability. It carries every
target, explicit knobs (an empty mapping is valid for a forkless anchor), paired positive `emmy_us` / reference
timings, the exact GPU identity and compute capability, and the immutable model ID. Validate it with repository-golden
validation and lower every entry again from the committed file. Keep it even when a target misses the bar or serving
fails later. A partial inventory never goes under `golden/`: keep it and its diagnostics outside the repository, and
never create an empty golden. If the model has no recipe directory, create an `onboarding`/`untested` shell first.

## 3. Create and validate a conservative baseline

Create or update the shared experiment recipe with a conservative feasible baseline — the smallest tensor-parallel size
that fits is a useful single-replica start, not necessarily the final datacenter strategy — keeping the current matrix
row exact and preserving other platforms' rows. Deploy before benchmarking:

```bash
emmy deploy ssh --recipe experiments/<model>/serving --ssh <target>
```

Before measuring performance, require: weights load and the server reaches health; a real chat or completion request
returns coherent output; advertised tool calling returns structured `tool_calls`; advertised reasoning lands in the
engine's reasoning field; requested multimodal input is exercised, or disabled for a text-only run; and the largest
claimed context is tested with an input that materially fills it. Start context testing at the native maximum; on an
out-of-memory or capacity failure, halve it and retry from a clean deployment. Never claim a context from startup
alone. Search upstream issues for an unfamiliar error and make one evidence-backed change per retry.

Run `emmy ... --teardown` after every failed deployment and before changing a serving configuration. Never mutate the
remote checkout or containers by hand; reading `nvidia-smi`, container state, and logs is fine.

## 4. Create the Emmy serving configuration

Do not infer Emmy support from a similar model. Emmy is **eligible** only when every gate passes on the requested
hardware and exact checkpoint quantization:

1. the live compute capability is accepted by Emmy's CUDA backend;
2. the architecture has a real trace and runner path in this checkout;
3. the checkpoint quantization has a matching Emmy loader and serving path that keeps the stored representation in
   the deployed graph (reference-only dequantization does not count);
4. section 2 committed a complete golden;
5. representative kernel correctness holds, and `emmy serve --runner generate` or the embedding path serves the checkpoint.

Record `eligible` or `ineligible` and the first failed gate in the report. An ineligible model can still get a vLLM or
SGLang recipe; the golden stays either way.

`EMMY_FAST_MATH=1` is the default candidate for every Emmy recipe. Compare it with standard Emmy on the exact
checkpoint, hardware, serving shape, capability probes, and a checkpoint-appropriate accuracy suite with a predeclared
tolerance, and select it unless a correctness, capability, or quality regression appears; then keep standard Emmy and
record the failed accuracy gate. FAST_MATH never changes eligibility, and a vLLM or SGLang recipe never sets it.

Run the `release-serving-image` workflow on the same server, following `docker/vllm-emmy-serve/ARCHITECTURE.md`. In
CI, the publication input replaces that workflow's conversational approval; every mechanical gate stays: golden
coverage, toolchain preflight, headroom sweep, HF parity, warm convergence, offline zero-recompile verification, and
the push. A failed gate means no push. A FAST_MATH recipe additionally needs its exact serving shape warmed to
convergence with a verified FAST_MATH pack hit and zero offline recompilation, and sets
`engine.llm.vllm.extra_env.EMMY_FAST_MATH: "1"`. Name the config and image with the repository slug helper, publish
only the verified tag as `cloudriftai/vllm-emmy-<model-slug>:<tag>`, pin the recipe to it, log in with
`docker login --password-stdin`, and run `docker logout` on success and failure.

When a gate fails, the recipe falls back to the vLLM or SGLang lane from section 3, and the report names the gate.

## 5. Measure the final serving configuration

Keep each benchmark variant under 20 minutes and setup plus model loading under 30. Past a cap, stop, halve the
request count or concurrency, tear down, and retry. Reserve at least 15 minutes of the deadline for artifacts and
cleanup, and start no stage that cannot finish in the remaining time.

Read [`prompts/onboard-model/benchmark.md`](../../../prompts/onboard-model/benchmark.md) completely and apply the lanes
relevant to this model, platform, and decision. Pick its consumer or datacenter profile and document rows that do not
apply or cannot finish. For an ordinary consumer deployment, the concurrency-1 measurement comes first. Let fit,
capacity, topology, latency needs, and measured behavior set concurrency, scheduler and memory settings, and
parallelism.

When Emmy qualified, the experiment compares the pinned vLLM or SGLang image with the pinned Emmy image on the same
model, GPU count, workload, context, request count, concurrency, warm-up, and precision; otherwise it runs only the
vLLM or SGLang lane, never a placeholder Emmy lane. Add only the comparisons that support this model's decision,
including a cache-reuse lane when caching matters for the intended deployment, and name the lane the final recipe
selects.

Run the final measurement with the `run-experiment` skill. Store it as this platform's archive, update this
platform's section of the experiment `RESULTS.md` including failed rows, and remove this platform's legacy top-level
records only after verifying the archive. Fold the winner into a single-variant `recipes/<model>/recipe.yaml`, which
has no `benchmark:` block.

## 6. Write the durable report

`recipes/<model>/RESULTS.md` is written by you, never by a script, and reads without the working directory.

**Beside a valid recipe**, it is the recipe's report, and updating a recipe without it is incomplete. Before writing a
number, find a successful experiment row and raw artifacts for the exact recipe configuration — model revision, image,
GPU name and count, precision, context, concurrency, workload, and engine knobs all match — and re-run the lane with
`emmy bench` when that evidence is missing or stale. Never estimate a value, copy a competing engine's result, or mix
runs. Update only this platform's section. Structure it for this model's evidence, with compact narrative and only
the tables that help, drawing from:

- date, repository revision, model revision, GPU name and count, driver, CUDA, pinned images;
- the exact workload; validated context, modality, tool-call and reasoning-parser results;
- throughput, TTFT, TPOT or ITL, failure count, and duration for the selected lane, plus any comparison that
  justified the choice;
- Emmy eligibility with its evidence; compiler coverage, tuned target counts, O3 verification, targets still slower
  than `torch.compile`, and remaining gaps; the golden's path when it is complete;
- for an Emmy recipe, the published image tag, kernel-tuning summary, and the accuracy result behind the FAST_MATH
  decision; for a vLLM or SGLang recipe, that engine's result without a comparison column;
- one `emmy bench experiments/<model>/<name> ...` reproduction command filtered to the selected lane;
- limitations and unresolved upstream issues.

**When no engine produces a valid recipe**, add a dated failure entry to that file instead — create it beside the
shell if absent, and keep earlier entries. Give the platform, the first failed gate, the evidence, what would unblock
it (for example, the engine release that adds support), and the golden's path when section 2 committed one. Add the
`onboarding-failed` tag to the recipe and change nothing else in it: nightly selection skips the recipe until an
explicit manual retry succeeds.

## 7. Verify and hand off

Before reporting success:

```bash
emmy bench --dry-run experiments/<model>/serving
emmy deploy ssh --dry-run --recipe recipes/<model> --ssh <target>
```

Also check:

- the coverage manifest accounts for every layer and seam; a complete trace has a committed golden whose every row
  has paired positive O3 and reference timings and lowers on the requested compute capability, and a partial trace
  has nothing under `golden/`;
- only repeated O3 rows are called deployable, and each tuning winner names its O1 ranking lane;
- a serving recipe has at least one successful result for its exact lane, and its report uses only numbers from that
  run;
- recipe and experiment pin immutable images and target exactly the requested GPU name and count;
- a published Emmy image passed offline zero-recompile verification, and a FAST_MATH recipe has a verified FAST_MATH
  pack for its serving shape; `EMMY_FAST_MATH=1` is in an Emmy recipe unless its accuracy gate regressed, and never in
  a vLLM or SGLang recipe;
- the experiment snapshot has the shared `recipe.yaml` and `RESULTS.md` plus this platform's LFS archive with its
  records inside, and other platforms' archives and sections are unchanged;
- every new corpus case reproduces its named stage, `pytest tests/compiler/realization` is green,
  `make test-corpus-regen` is a no-op, and every `_xfail_*` note has an `evidence:` paragraph;
- nothing staged is a dated run directory, loose benchmark output, or onboarding summary, and no tracked artifact holds
  credentials, absolute scratch paths, or VM identifiers;
- workloads are torn down and `docker logout` has run (`cleanup.docker_logout: true` also covers a stock-only run that
  never logged in).

Write the summary atomically to the caller's path outside the repository and print that path as the last line. It
carries the requested mode on success and failure. `artifacts` lists every repository file the run created, modified,
or deleted, including bounded fixes, their tests, and every corpus case (a case is staged only when listed here, and is
also reported in `compiler.realization_gaps`). `compiler_artifacts` lists only a complete golden.
`experiment_artifacts` lists only the shared experiment recipe and report and this platform's archive; this
platform's deleted legacy records go in `artifacts` only. Never list other platforms' unchanged files or the dated run
directory. On success, `deployment_summary` names the selected engine, image, and key serving settings, and
`performance_summary` gives the selected lane's workload and key throughput, latency, and failure numbers, each one
line of at most 1000 characters from that exact lane; the workflow notification uses them.

```json
{
  "status": "success",
  "mode": "onboarding",
  "model_id": "org/model",
  "target": {"gpu": "exact hardware.py name", "gpu_count": 1, "ssh": "user@host"},
  "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
  "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, p50 TPOT 7.1 ms, 0 failures",
  "recipe": "recipes/<model>/recipe.yaml",
  "experiment": "experiments/<model>/serving/recipe.yaml",
  "artifacts": [
    "recipes/<model>/recipe.yaml",
    "recipes/<model>/RESULTS.md",
    "recipes/<model>/golden/<gpu-slug>_<compute-cap>.json",
    "tests/compiler/realization/cases/matmul/<name>_xfail_offered.json",
    "experiments/<model>/serving/recipe.yaml",
    "experiments/<model>/serving/RESULTS.md",
    "experiments/<model>/serving/results_<gpu-short>x<gpu-count>.tar.gz"
  ],
  "compiler_artifacts": [
    "recipes/<model>/golden/<gpu-slug>_<compute-cap>.json"
  ],
  "experiment_artifacts": [
    "experiments/<model>/serving/recipe.yaml",
    "experiments/<model>/serving/RESULTS.md",
    "experiments/<model>/serving/results_<gpu-short>x<gpu-count>.tar.gz"
  ],
  "report": "recipes/<model>/RESULTS.md",
  "compiler": {
    "coverage": "complete",
    "golden": "recipes/<model>/golden/<gpu-slug>_<compute-cap>.json",
    "traced_targets": 42,
    "tuned_targets": 42,
    "blocked_paths": [],
    "realization_gaps": [
      {"file": "tests/compiler/realization/cases/matmul/<name>_xfail_offered.json",
       "stage": "offered", "emmy_us": 30.81, "tcompile_us": 24.10}
    ]
  },
  "emmy": {"eligible": true, "reason": "all eligibility gates passed", "image": "cloudriftai/...:tag"},
  "cleanup": {"workloads": "complete", "docker_logout": true},
  "failure": null
}
```

When no valid recipe qualifies, write `status: "failed"`, null `recipe`, `experiment`, `deployment_summary`, and
`performance_summary`, and empty `experiment_artifacts`. `report` still names `recipes/<model>/RESULTS.md` with the
failure entry, and `artifacts` lists it with the tagged recipe and the golden, corpus cases, and bounded fixes the run
keeps — never experiment or image files. `failure` has `gate`, a concise credential-free `message` (it goes to Discord),
and `regression`. `regression: true` means a previously qualified behavior or measured lane no longer meets its prior
contract and the bounded fix could not restore it; a model that never qualified is a failure, not a regression.
Diagnostics and partial inventories stay outside the repository. Tear down workloads, never claim partial onboarding
as success, and leave the VM to the caller.
