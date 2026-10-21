# GDN performance audit across cards and quantizations (2026-10-21)

Where Emmy's Gated DeltaNet (GDN) programs are fast and where they are slow, by card and by quantization, measured
against `torch.compile`, with the gaps closed where a record or a small compiler fix closes them. Two agents run the
card tasks below in parallel; this file is the task list and the place each agent's state lands until the report is
written and the file is deleted.

## What exists

GDN programs come from two model families. Qwen3.8-27B is dense, 48 of 64 layers GDN, hidden 5120; its serving twins
are one whole-layer program per static width (`gdn1`, `gdn16`, `gdn64`) beside the attention `pre`/`post` twins.
Qwen3.6-35B-A3B is a MoE hybrid served with a two-token MTP draft, so its twins add the per-row verify programs
(`gdn2rows`, `gdn3rows`, `gdn4`). Every GDN whole-layer target fuses into one kernel offering dozens of cut seams,
and the cut pass prices every arm by building it; a compile no row decides takes minutes per target (an hour or more
when the prior picks a single-CTA arm over the big contraction).

| Golden | Card | Quantization | GDN targets | State on `main` |
| --- | --- | --- | --- | --- |
| `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` | V100 SXM2 16GB | FP8 weight-only, FP16 trunk | gdn1 25.2 ms, gdn16 10.9 / 7.7 / 1.4 ms, gdn64 41.0 / 7.5 / 1.5 ms | tagged `prior-pending`; pre64 9.4 ms vs 0.69 ms `torch.compile` |
| `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` | RTX 5090 | NVFP4 W4A4, BF16 trunk, fast-math lane | gdn1 217 us, gdn16 170 us (tc 87), gdn64 344 us (tc 103) | serving-qualified; GDN roots 2-3x behind `torch.compile` |
| `recipes/Qwen3.6-35B-A3B-FP8/golden/rtx5090_sm120.json` | RTX 5090 | FP8 weight-only, BF16 trunk | gdn1 1.04 ms, gdn2rows 630 us, gdn3rows 179 us, gdn4 2.79 ms, gdn64 236 / 194 us | gdn4 and gdn3rows hold the prior's pathological cut picks |
| `recipes/Qwen3.8-27B-{EXL3,GPTQ-Int4,AWQ-INT4}/golden/v100_sm70.json` | V100 SXM3 / SXM2 / SXM3 | EXL3, GPTQ-Int4, AWQ W4A16 | one decoder-layer program each, a handful of rows | compiler-qualified only |

No whole-layer GDN target has an eager or `torch.compile` time in any golden: the GDN state wrapper writes its next
state as a `ScatterOp`, and `torch_ref` (`emmy/compiler/backend/torch_ref.py`) maps no `ScatterOp`, so the target has
no Torch twin and benches Emmy alone. The sub-targets with a time (`k_matmul`, `k_cumsum`) are pieces, not the layer.

The `ScatterOp` mapping is the first task: it is the mirror of the `GatherOp` mapping already there (clone the
destination, `scatter` for a same-rank index, `index_copy` for a row index, an index equal to the axis extent writes
nothing). Without it the audit cannot answer its own question.

## Cards

| Card | Capability | Checkout | Notes |
| --- | --- | --- | --- |
| H100 80GB (GCP) | sm90 | `~/emmy-gdn-audit` | CUDA 13.0 at `/usr/local/cuda`, 26 cores, 230 GB |
| A100 40GB (GCP) | sm80 | `~/emmy-gdn-audit` | CUDA 13.0, 12 cores, 83 GB |
| RTX 5090 (CloudRift) | sm120 | `~/emmy-gdn-audit` | CUDA 13.3; 23 GB RAM, one compile at a time; previous session's files under `~/q36` |
| RTX 4090 (CloudRift) | sm89 | `~/emmy-gdn-audit` | CUDA 12.9, 18 cores, 78 GB |
| V100 SXM2 16GB (CloudRift) | sm70 | `~/emmy-gdn-audit` | CUDA 12.9, torch cu126, 6 cores, 50 GB |
| V100 SXM3 32GB (CloudRift) | sm70 | `~/emmy-gdn-audit` | CUDA 12.9, torch cu126, 4 cores, 83 GB |

Every host has the audit branch pushed into `~/emmy-gdn-audit` (git, `receive.denyCurrentBranch updateInstead`) and
its own `venv` from `make setup`; `~/emmy-gdn-audit-setup.log` ends with `SETUP_DONE`. Non-interactive ssh shells
have no CUDA on `PATH`: every remote command is prefixed
`PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda NUMPY_MADVISE_HUGEPAGE=0`.

## Lanes

A lane is one model at one quantization. The matrix is lanes x cards; a cell is the GDN twins of that lane traced for
that card, benched, and recorded.

| Lane | Model | Twins from | Cards |
| --- | --- | --- | --- |
| A | Qwen3.8-27B-FP8 | the V100 SXM2 golden on its card; `emmy trace --serving-twins` elsewhere | V100 SXM2 (golden), V100 SXM3, 4090, A100, H100 |
| B | Qwen3.8-27B-NVFP4 | the 5090 golden | 5090 only (W4A4 is sm120) |
| C | Qwen3.6-35B-A3B-FP8 | the 5090 golden; a trace on the H100 | 5090 (golden), H100 |
| D | Qwen3.8-27B unquantized (BF16; FP16 on Volta) | `emmy trace --serving-twins` with a new `serving-<card>.env` | H100, V100 SXM3 |
| E | Qwen3.8-27B EXL3 / GPTQ-Int4 / AWQ-INT4 layer goldens | their V100 goldens | V100 SXM3 / SXM2 / SXM3 |
| F (optional) | Qwen3.5-9B, one GDN layer through `emmy run MODEL --layer N --bench --bench-backends eager,tcompile,emmy` | the HF layer itself | every card, if the single-layer trace works for a GDN layer |

Lane F is the one probe where the reference is the HF module under `torch.compile` rather than Emmy's Torch twin of
the traced program; it is optional and capped at ten minutes of trying per card.

## Per cell

1. **Bench what the card deploys.** On the golden's own card: `emmy run --golden FILE --bench --json DIR` over the GDN
   realizations (`--realization gdn1` etc.; the whole file includes the attention twins). On a new card: write
   `recipes/<model>/serving-<card>.env` from the existing one (SERVE_GPU, dtype, no pipeline flag), trace
   `emmy trace MODEL --serving-twins --serving-config ENV -o _tune/gdn/<card>/working.json`, then the same bench.
   Record per target: Emmy us (the greedy pick), `torch.compile` us, eager us, the accuracy verdict, compile time, and
   whether `--strict-evidence` deploys it. `EMMY_KERNEL_TIMEOUT_MS=20000` keeps a slow prior pick from counting as
   a hung kernel.
2. **Classify the gap.** Behind `torch.compile` by a factor, or a multi-millisecond absolute time: say which pieces
   (the `--json` kernel table) and whether the pick is the prior's (no row) or a measured row.
3. **Close what a record closes.** `--record-greedy` the targets with no rows (strict: pin the route with
   `--pin-route` or `EMMY_KNOBS` where a fork has no row), `--kernel PIECE --tune N --bench` the slow pieces, re-record,
   and promote into `recipes/<model>/golden/<card>.json`. The refresh-golden skill, steps 3 and 4, is the flow. Never
   record a row whose accuracy check failed, and never re-record to make a red node green.
4. **Fix what a small compiler change fixes.** A missing lowering, a wrong pick a guard would catch, a bench that
   refuses a valid target. Own commit, with a test, under about sixty lines; anything larger is a finding in the
   report, not a fix. Fusion stays maximal and normalization carries no knob (AGENTS.md).
5. **Strict audit** where a serving env exists: `emmy eval golden --golden FILE --serving-config ENV`.

## Assignment

- **Agent DC** (H100, A100, 5090): the `ScatterOp` Torch twin first, verified on a 5090 GDN target; lane B and
  lane C on the 5090; lane A on the H100 and A100; lane D (BF16) and lane C on the H100; lane F if cheap.
- **Agent Volta** (V100 SXM2, V100 SXM3, 4090): lane A on the V100 SXM2 golden first (the 25-41 ms rows), then lane E,
  lane A on the 4090 and V100 SXM3, lane D (FP16) on the V100 SXM3; lane F if cheap.

Each agent commits on its hosts' clones, on this branch, one card's files per commit (goldens and env files are per
card, so the merge is trivial); compiler fixes in their own commits, announced in the agent's `fixes.md` in the
session scratchpad before the edit so the two agents never fix the same thing twice. Nobody edits the dev-box tree.

## Report

One table per lane, cards as columns, one row per GDN target: Emmy us, `torch.compile` us, ratio, strict status,
what changed in this audit (recorded, tuned, fixed, open). Then the analysis: which gaps are the prior's picks (a
record closes them), which are lowering gaps (a compiler change), which are the card's (Volta without BF16 or FP8
MMA), and which are the quantization's (the FP8 or FP4 decode in the fused kernel). The PR body carries the report;
this file is deleted when it lands.
