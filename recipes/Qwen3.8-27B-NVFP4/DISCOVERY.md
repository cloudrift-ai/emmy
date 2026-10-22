# Qwen3.8-27B-NVFP4 discovery

## Measured serving performance (2026-10-21, shared GDN recurrence output)

The cut pass now treats the GDN recurrence output as one value read flat, by blocks of sixteen and by packed pairs,
so its three readers share one workspace. `golden/rtx5090_sm120.json` was re-recorded for the width-1, width-16 and
width-64 GDN programs (fast-math lane). Per program, greedy under `--strict-evidence` with the golden alone:

| Program | Kernels | Sum of kernels µs | Main (#1210) µs |
|---|---:|---:|---:|
| gdn1 | 12 | 98.0 | 106.6 |
| gdn16 | 15 | 96.5 | 205.0 |
| gdn64 | 15 | 121.2 | 262.2 |

Part of the width-16 and width-64 gain is tuning, not the shared output: one piece with the same identity on main runs
45.3 µs there and 2.6 µs tuned. The strict serving audit deploys 9 of 9 fast-math twins, as on main; the standard
lane, which serving does not use, deploys 0 of 9 (main: 1).

Serving uses the setup of 2026-10-20: one RTX 5090, one request, a BF16 trunk, decode bucket 16, prefill chunk 64, M1
tier 0, fast math, no prefix cache, vLLM 0.23.0 (torch 2.11.0, Transformers 5.14.1), `--strict-evidence`, an empty
tune DB. The strict boot took 906 s with an empty cubin cache (846 s on 2026-10-20). Prompts tokenize to the same ids.

| Formatted prompt tokens | Output tokens | Emmy decode tok/s | Emmy TTFT s | 2026-10-20 decode tok/s | 2026-10-20 TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 | 57.2 (57.2–63.9) | 0.16 (warm) | 55.9 (55.9–58.7) | 0.17 (warm) |
| 227 | 39 | 57.2 (57.2–60.0) | 0.22 (0.22–0.37) | 55.9 (55.9–58.8) | 0.25 (0.25–0.28) |
| 1027 | 219 | 57.2 (57.2–57.4) | 0.64 (0.64–0.65) | 56.1 (56.1–56.5) | 0.74 (0.74–0.75) |

One boot, two probe rounds; the first value is the second round, ranges are min–max over both. Decode is about 2%
faster and the first token comes 10–14% sooner on the medium and long prompts. All answers are correct. The medium
and long answers are worded differently from 2026-10-20: the long one now walks through the counts before giving the
same branches and difference (74), in 219 tokens instead of 82.

## Measured serving performance (2026-10-20)

The tuned `golden/rtx5090_sm120.json` deploys all nine serving programs of the fast-math lane under strict evidence
(`emmy eval golden --serving-config recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env`: 9 of 9 fast-math twins; the
standard lane, which serving does not use, deploys 1 of 9). Setup as on 2026-10-16: one RTX 5090, one request, a BF16
trunk, decode bucket 16, prefill chunk 64, M1 tier 0, fast math, no prefix cache, vLLM 0.23.0 (torch 2.11.0,
Transformers 5.14.1). Emmy boots with `--strict-evidence` and an empty tune DB; the strict boot took 846 s. Stock
vLLM runs the same flags with compiled CUDA graphs. Both sides tokenize the probe prompts to the same ids.

| Formatted prompt tokens | Output tokens Emmy / stock | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 / 8 | 55.9 (55.9–58.7) | 0.17 (warm) | 63.2 (60.2–63.3) | 0.08 (warm) |
| 227 | 36 / 36 | 55.9 (55.9–58.8) | 0.25 (0.25–0.28) | 63.3 (63.3–63.3) | 0.30 (0.30–0.43) |
| 1027 | 82 / 196 | 56.1 (56.1–56.5) | 0.74 (0.74–0.75) | 63.2 (63.2–63.2) | 1.24 (1.24–1.24) |

One boot per side, two probe rounds; the first value is the second round, ranges are min–max over both. All answers
are correct; the short and medium answers are identical on both sides, and the long one gives the same branches and
difference (74) in fewer words on Emmy. Decode is 12% below stock; time to first token is lower than stock's on the
medium and long prompts.

GSM8K (`scripts/run_lmeval_gate.py`: 200 questions, 5-shot, seed 0, raw completions, the protocol of
`experiments/gemma-4-12B/gsm8k_rtx5090`), strict match: Emmy 0.450, stock 0.470, standard error 0.035 each; flexible
extract: Emmy 0.535, stock 0.530. The two sides agree within one standard error. Both absolute scores are low for this
model, likely because the protocol sends raw completions to a chat model.

## Measured serving performance (2026-10-16)

This compares main 7ef17a58 with PR #1171 at 7025e8b4 on one RTX 5090. The setup is the 2026-10-13 one: each
side's own `golden/rtx5090_sm120.json`, one request, a BF16 trunk, decode bucket 16, prefill chunk 64, M1 tier 0,
fast math, no prefix cache, vLLM 0.23.0 (torch 2.11.0, Transformers 5.14.1), eager execution. Each side has its own
cubin cache. The prompts, the probe and the throughput formula are those of 2026-10-13.

Main boots with `--strict-evidence`. The branch does not: its strict boot stops after about 190 s, twice, at the
first width-16 GDN program:

```
EvidenceError: strict evidence: kernel 'k_linear_matmul_mean_reduce_d29945' (node 'add_70') has no measured
evidence for its 030_cut fork (no measured row spells a kernel-set arm)
```

The strict serving audit (`emmy eval golden --serving-config recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env`) agrees.
With the branch golden, 1 of 9 twins deploys from the file's rows in each lane; only gdn1 does. With main's golden,
all 9 do. The branch golden has 146 kernels and 294 rows; main's has 236 and 524. Six traced roots of the branch
golden have no row at all, so `emmy run --record-greedy` cannot record them: it implies strict evidence and stops
at the same fork. The branch numbers below therefore come from a boot without `--strict-evidence`, where the prior
picks every fork no row decides.

| Formatted prompt tokens | Output tokens main / branch | Main strict decode tok/s | Main strict TTFT s | Branch non-strict decode tok/s | Branch non-strict TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 / 8 | 32.2 (31.7–34.3) | 0.60 (warm) | 14.0 (14.0–14.1) | 1.11 (warm) |
| 227 | 38 / 38 | 31.9 (31.7–32.3) | 1.07 (1.02–1.17) | 14.1 (14.0–14.1) | 18.5 |
| 1027 | 223 / 208 | 31.7 (31.6–31.7) | 1.51 (1.51–1.52) | 14.0 (14.0–14.0) | 89.3 |

Main is two strict boots with two probe rounds each; the branch is one boot with two rounds. Ranges are min–max.
Strict main boots took 897 s with an empty cubin cache and 792 s with a warm one. The non-strict branch boot took
877 s. The first request after main's cold boot waited 58.7 s for its first token; later boots answered it in
under 2 s. All answers are correct. The short and medium answers are identical on both sides. The long answer
differs in wording and gives the same branches and difference (74).

So this measurement does not show the PR's kernel gains in serving. The branch serves at less than half of main's
decode rate, with a much slower prefill, because the prior decides the forks the restamp left without rows.

## Measured serving performance (2026-10-13)

The full model serves with the recipe's `golden/rtx5090_sm120.json` and `--strict-evidence`. The measured setup is
one RTX 5090, one request, a BF16 trunk, decode bucket 16, prefill chunk 64, M1 tier 0, fast math, no prefix cache,
and vLLM 0.23.0. Emmy uses eager execution; stock vLLM uses compiled CUDA graphs. Strict server boot took 646 s.
Stock vLLM was not run again; its columns are the 2026-10-09 measurement.

| Formatted prompt tokens | Output tokens | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 | 29.5 | 0.61 (warm) | 63.5 | 0.69 |
| 227 | 36 | 29.2 | 1.05 | 63.0 | 0.29 |
| 1027 | 229 | 29.2 | 1.19 | 62.9 | 1.18 |

These are single streaming requests with natural EOS. Decode throughput is `(output tokens - 1) / (last - first)`;
time to first token (TTFT) is measured from request submission. All three answers are correct. The short and
medium answers match the 2026-10-09 ones token for token; the long answer is worded differently and gives the same
facts. Evidence:
[the measurement report on PR #1160](https://github.com/cloudrift-ai/emmy/pull/1160#issuecomment-6150058296).

The gated DeltaNet layers take a prompt in chunks of 64, 16 and 1 tokens, widest first, so each prompt length
weighs the three widths differently. Against 2026-10-09, the width-16 program is about twice as slow, because its
conv kernel recomputes the qkv projection once per conv tap; the width-64 program is about 20% faster; and the
width-1 program, which also runs decode, is 7–10% faster. That makes decode faster at every length, the 227-token
prompt (two width-16 chunks) slower to its first token, and the 1027-token prompt (no width-16 chunk) faster.

### Previous measurement (2026-10-09)

Same setup with the golden of that date. Strict server boot took 523 s.

| Formatted prompt tokens | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|
| 25 | 26.4 | 0.49 (warm) | 63.5 | 0.69 |
| 227 | 26.3 | 0.71 | 63.0 | 0.29 |
| 1027 | 26.2 | 1.47 | 62.9 | 1.18 |

Evidence: the measurement reports in
[PR #1120's discussion](https://github.com/cloudrift-ai/emmy/pull/1120#issuecomment-6078603008).
The earlier full-Emmy strict boot of the first golden decoded at about 0.13 s per token
([PR #1023](https://github.com/cloudrift-ai/emmy/pull/1023), 2026-10-04).
[PR #993](https://github.com/cloudrift-ai/emmy/pull/993) was closed; its mixed route ran only the MLPs in Emmy
and measured against eager vLLM, so that comparison does not apply to the compiled-graph baseline above.

## Assessment on 2026-10-05

This historical assessment retained a serving-blocked label; full-model serving was enabled by
[PR #1023](https://github.com/cloudrift-ai/emmy/pull/1023). The measurements above supersede that label.

The recipe was assessed as best-effort (compiler-qualified, serving-blocked), heat 55. The ecosystem was strong:
the base Qwen/Qwen3.8-27B held 6,821,761 30-day downloads and 16,960 likes
(https://huggingface.co/Qwen/Qwen3.8-27B), and the verified NVFP4 nodes carry the bulk of the quant volume
(unsloth/Qwen3.8-27B-NVFP4 at 2,365,557 30-day downloads,
https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). The serving-blocked tag meant engine support for the NVFP4 path was
not yet qualified in that assessment; the specific Inferact checkpoint also did not resurface in its bounded
identity checks (bounded search, not deletion evidence).
