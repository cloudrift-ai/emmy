---
name: discover-models
description: >-
  Use this skill when the user asks what new models to try or benchmark, wants newly released open models discovered,
  wants trending models mapped to suitable GPU hardware, or wants the maintained recipe set refreshed. It produces a
  ranked shortlist or lifecycle selection ready for the `onboard-model` skill, using keyless discovery data, web
  search, and a VRAM fit calculation.
---

# Discover Models to Explore

Turn "what models are worth our GPU hours?" into a shortlist or a refresh of Emmy's recipe lifecycle. New open-weight
models are filtered to the ones with real demand and mapped to the GPU platforms they fit. Every recipe gets a 0-100
heat score for current onboarding priority; the maintained set stays small and focused, other useful recipes stay
best-effort, and only technically superseded or unusable models become obsolete.

Research is keyless and read-only: public OpenRouter and Hugging Face endpoints plus web search. No server is touched
and no recipe is edited — the workflow applies a validated lifecycle manifest, and `onboard-model` owns deployment
and qualification. In automated mode, the agent may create or edit only recipe-local `DISCOVERY.md` notes to retain
research behind its decisions.

```
HF/OpenRouter data ─┐
Reddit discussions ─┼→ reconcile exact models → score heat → VRAM fit → GPU platform → hardware/model matrix
OpenRouter/Arena ───┘
```

## Choose the mode

**Automated lifecycle mode** (GitHub Actions): read `prompts/discover-models/lifecycle.md` and
`prompts/discover-models/score-recipes.md` and the recipe-local `DISCOVERY.md` notes supplied in the task before
research. `lifecycle.md` is the whole contract — task payload, delegation to the source, scorer, and fit subagents,
selection rules, and output JSON — and this skill adds only the background below. Ask no questions, never rebuild the
inventory the task supplies, and return the selection as soon as the evidence supports it. The workflow runs the agent
on a checkout of `main` and commits the validated manifest and any changed research note itself; the agent never
touches git.

**Survey mode** (interactive): produce a shortlist and a hardware/model matrix. Ask only what the user has not
implied:

- **Time window** — default the script's ~90 days; widen with `--since 2026-01-01`.
- **Modality** — include multimodal by default; `--text-only` if they serve only text.
- **Target hardware** — default every GPU in `emmy/gpu.py`; bucket only for a subset they name.
- **Finalists** — default 5-8, spread across hardware tiers.

## Recipe lifecycle

- `maintained` — tested and selected for periodic testing and optimization;
- `best-effort` — runnable and useful, outside the periodic set;
- `obsolete` — kept for history but disabled;
- `onboarding` + `untested` — a shell not yet onboarded; scored, never classified.

A person can add `lifecycle-locked` beside any lifecycle tag. The workflow then leaves the recipe out entirely: it is
not scored, not classified, and not counted against the maintained set.

Untagged complete recipes are classified on the first lifecycle run. Obsolete is a conservative, tradeoff-free
decision: a named replacement for the same task whose smallest deployment uses no more total GPU memory than the old
recipe's smallest, with no material advantage left to the old model in quality, context, hardware, latency,
throughput, cost, modality, or licensing — or a concrete technical reason the recipe should no longer be used. Low
demand, age, a larger sibling, and exclusion from the maintained set are not reasons. Prefer best-effort whenever the
evidence is ambiguous. An obsolete recipe can return when the evidence changes. Read a complete recipe only for a
specific obsolete comparison.

## 1. Collect candidates

In survey mode, run the discovery script with arena enrichment, JSON for parsing and the table for a human view. Keep
`--workers 4` and do not re-run in a loop: Hugging Face rate-limits bursts, and transient misses land in its "COULD
NOT VERIFY" bucket.

```bash
./venv/bin/python scripts/new_models.py --arena --workers 4 --json > /tmp/new_models.json
./venv/bin/python scripts/new_models.py --arena --workers 4          # readable table
```

It lists open-weight models OpenRouter hosts, excludes active families already in `recipes/` (obsolete ones may
resurface as reactivation candidates), drops anything older than `--since`, verifies each on Hugging Face, and ranks by
momentum. `--help` lists every flag. Each JSON row in `models[]` carries:

| Field | Meaning | Use |
|---|---|---|
| `hf_id` | Hugging Face repo ID | the model identity; feeds `onboard-model` |
| `created_at` | release date | recency |
| `downloads` | 30-day pulls | adoption; lagging and biased toward small models |
| `likes` | cumulative likes | reputation |
| `trending` | trendingScore | momentum — the best single demand signal |
| `elo` / `arena_rank` | LMArena Elo and rank | quality; blank usually means too new, not bad |
| `modality` | `text->text`, `text+image->text`, … | multimodal flag |

The table footer flags stale OpenRouter→HF mappings ("NOT ON HF") and likely arena name mismatches; a mismatch can
hide a strong Elo. Take the top 8-12 by `trending` (then `elo`, then `downloads`).

In automated mode the agent cannot run scripts; the Hugging Face and OpenRouter source subagents supply the same
public catalog evidence.

## 2. Check independent demand

Reddit is an independent source, not only a check on script results: read recent high-engagement threads and merge
their models with the catalog candidates. Accept a Reddit-only model only once its exact open-weight Hugging Face ID
is resolved. For each top candidate, search:

- `"<model>" release` / `"<model>" benchmark` — announcements and benchmark claims (MMLU, GPQA, LiveCodeBench,
  SWE-bench, AIME);
- `"<model>" vs` — head-to-head comparisons, a sign people care;
- Reddit (r/LocalLLaMA), Hacker News, X — is it discussed, or did it land silently;
- the lab's track record — established labs draw adoption faster.

Give each a one-line demand read: *strong* (benchmark wins, active discussion, reputable lab), *moderate*, or
*niche/quiet*. High trending and loud online is a strong pick; high downloads but silent is often a small fine-tuning
base. Score heat with the bands in `prompts/discover-models/score-recipes.md`, and compare all scores within the run.

## 3. Shortlist

A model is promising when it scores on several of: high `trending`; high `elo`; strong demand read; and novelty for
Emmy — a new architecture, quantization, or size teaches more than another sibling of an existing recipe. Drop tiny
fine-tuning bases riding download counts and anything the user does not care about. Aim for a spread of sizes across
hardware tiers.

## 4. Size each finalist

Follow [`prompts/model-fit.md`](../../../prompts/model-fit.md), the fit contract `onboard-model` shares: total
parameters from `config.json`, a quantization whose repository exists, and the stated arithmetic. Canonical GPU names
and `vram_mib` come from `emmy/gpu.py`. A model no fleet platform can hold gets no deployment. In automated mode this
is the `discover-fit` subagents' job, one per new candidate, under `prompts/discover-models/size-deployments.md`.

## 5. Deliver

In survey mode, present a hardware/model matrix — each GPU platform with the promising models that fit, the
quantization, a one-line reason, and the fit numbers:

| Hardware | gpu_count | Recommended model (quant) | Why it's promising | Fit note |
|---|---|---|---|---|
| RTX 4090 / 5090 | 1 | `<8B model>` (AWQ/FP8) | strong small-model Elo, hot on HF | ~X GB, fits one card |
| H200 141GB | 1 | `<120B>` (FP8) | flagship, loud online | fits 1×, long context |
| B200 | 8 | `<400B+ MoE>` (FP8/NVFP4) | top open model, high demand | TP8 across the node |

Mark a model with no engine support or no suitable quantization "watch, revisit" instead of slotting it. Then offer to
run `onboard-model` for each pair the user wants, passing the `hf_id`, GPU, and `gpu_count`.

In automated mode, return exactly the JSON `lifecycle.md` defines. The workflow validates it, restores existing
onboarding shells, derives best-effort recipes, and demotes an obsolete proposal to best-effort unless its
replacement is active, serves the same task, and passes the memory rule above.

## Common mistakes

- Ranking by downloads alone — `trending`, `elo`, and the demand read together are the signal.
- Assigning a platform from the model name instead of `prompts/model-fit.md`.
- Treating a blank arena Elo as bad.
- Hammering the script instead of waiting and re-running once.
