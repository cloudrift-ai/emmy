# RTX 5090 Qwen3.8 NVFP4 MLP tuning log

Status: bounded hand sweep and strict golden serving qualified on October 1, 2026; repository prior refit and CI
are in progress. This log belongs to the experimental mixed serving lane in
PR #993. It records measured choices separately from ideas borrowed from earlier plans and golden files. The exact
implementation and qualification boundary is in [the mixed serving progress report](nvfp4-qwen-mixed-serving-progress.md).

## Fixed target and method

The model is `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`, vLLM 0.23, BF16,
TP1/PP1, one RTX 5090 with driver 580.178.04. Emmy owns only the 64 text MLPs; stock vLLM owns attention, GDN,
state, and scheduling. Decode uses a static M=16 bucket and prefill a static M=64 bucket. Both permit fewer active
rows. The source helper `scripts/qwen38_nvfp4_mixed_5090_knobs.sh` pins `FAST_MATH=true`, five parent workspace
cuts, two per-shape child output cuts, and scoped native FP4 schedules for three contraction kernels. Every other
decision family has an explicit fallback value. The serving launcher now loads the measured model golden with
`--strict-evidence` and a fresh private tune DB per boot; it refuses inherited hand pins. No candidate is
accepted from an unpinned prior pick.

The user requested a bounded manual sweep because automatic prior tuning is broken. The first pass uses existing
`emmy run --ir --bench --bench-backends emmy --warmup 5 --iters 12` on the saved exact layer-0 loop-fusion IR,
with the sourced recipe's global and shape-specific `EMMY_KNOBS` combined. The command fails before compile if
either required variable is empty or a pin cannot be audited. Timings are warmed single-layer diagnostics,
not serving latency. For routine candidates, check every intended pin, native FP4 instruction, finite output,
and same-input agreement with the qualified baseline. Finalists get an independent stock quantized MLP oracle
before paired endpoint measurements. The isolated IR and logs are on the dedicated
host under `/root/emmy-nvfp4-serving/src-merged/postmerge-probe-{16,64}` and `logs/tune-*`.

Inside the pinned `emmy-vllm023-dev:local` container with that checkout mounted at `/workspace`, the audited
baseline command is:

```sh
set -euo pipefail
source scripts/qwen38_nvfp4_mixed_5090_knobs.sh
test -n "$EMMY_KNOBS" && test -n "$EMMY_MLP_STATIC_KNOBS"
export EMMY_KNOBS="$EMMY_KNOBS,$EMMY_MLP_STATIC_KNOBS"
emmy run --ir /workspace/postmerge-probe-16/04_loop_fusion.json \
  --bench --bench-backends emmy --warmup 4 --iters 8 \
  --json /workspace/tune-audit16-final.json
```

For M=64, substitute `EMMY_MLP_PREFILL_KNOBS`, `postmerge-probe-64`, and a distinct JSON path. The run log must
have no `unreproducible pin` line; inspect all seven JSON `greedy.kernels[].knobs` rows before accepting timing.
Manual trials change one scoped key in the composed string at a time and keep the remainder identical.

## Measured starting point

| Shape | Whole MLP CUDA graph | Native gate/up pieces | Native down | Other four kernels |
| --- | ---: | ---: | ---: | ---: |
| M=16 | 165.7 µs | 35.8 + 59.9 µs | 31.6 µs | 5.7 µs summed |
| M=64 | 398.3 µs | 159.2 + 152.7 µs | 75.7 µs | 9.4 µs summed |

The M=16 table came from `tune-run-ir-baseline16.json` with twelve iterations and five warmups; M=64 from
`tune-run-ir-baseline64.json` with the same settings. Kernel medians sum to 132.9 and 397.0 µs respectively; the
whole-program CUDA graph measurement is separately timed and can differ. The three contractions account for about
96% and 98% of summed kernel time. Tuning therefore starts with their WORK/TILE/STAGE choices. Scaling these
single-layer times to all 64 MLPs is only a rough inference: actual vLLM TPOT is 104.23 ms, so a kernel-only gain
may have a modest end-to-end effect.

Both baselines emit native `mma.m16n8k64` FP4 bodies. The M=16 first gate/up-derived kernel has 272 blocks,
64 threads, 45 KiB shared memory, 114 registers and 8% reported occupancy; the second has 1088 blocks,
32 threads, 15 KiB, 70 registers and 12% occupancy. The M=64 counterparts have 4352 and 2176 blocks,
32 threads, 15/25 KiB, 74/106 registers and 12%/8% occupancy. The down kernels have 160/640 blocks,
64 threads, 15 KiB, 54 registers and 25% occupancy. These observations suggest bounded WORK geometry and
stage-depth trials before tuning the 1–4 µs encode/scales kernels.

The first `emmy run --ir` pass flagged the current recipe as `unreproducible pin` even though scoped native rows
were realized. Two audit issues were identified: child-site `PLACE@place_<token>/...` receipts were stored under
their local site names, and a bare family fallback was compared against kernels with an overriding scoped pin.
The shared audit now records exact applied child source key/value receipts after choosing the cut and checks bare
schedule pins only on kernels without an overriding scoped pin. Focused tests reject stale/mismatched keys and
accept applied cuts and fuses. The audit then exposed a genuine unused `WORK=w1x4` fallback: all three WORK
choices were scoped to native contractions; the four encode/scales kernels stamp WORK OFF. The fallback is now
explicitly `WORK=`. Both shapes pass the fail-closed pin audit, and their seven schedules and resources match the
old baseline. New eight-iteration/four-warmup checks measured M=16 167.4 µs and M=64 399.9 µs whole-program
latency, consistent with the provisional baseline. One failed audit attempt accidentally sourced no pins due shell
quoting and ran a 1.19-second scalar route. It is excluded from all comparisons; the reusable knobs file now
prevents that failure. This audit fix is a measurement gate, not a numerical or performance fix.

## Candidate sources and bounded sweep

The older [Qwen performance plan](nvfp4-qwen-performance.md) and its `pins-cannot-target-one-piece.md` report
suggest native FP4 `f1x2/k4` and `/k8`, async depth 1–4, and WORK widths around `w1x1` to `w1x4` for decode.
Those are historical hypotheses, not results on this graph. The report's older scoping limitation is partly
addressed by this PR's generic child-site PLACE pin, but each final piece identity must still be rechecked after
IR changes. The hardware 5090 golden `emmy/compiler/pipeline/search/golden/records/rtx5090_sm120.json` has
FP16 MLP rows with larger N fragments and TMA depth 2–4; Gemma 4 and OLMoE 5090 model goldens also suggest
WORK layouts and `t128`–`t512` encode rows. Their FP16 MMA atoms cannot be copied into NVFP4 W4A4, so only
geometry and transport choices are candidates. The existing native FP4 realization cases provide legality
examples. TMA `/p2` is a bounded alternative to async, not an assumed improvement; old #971 notes it was even
or slower in related decode/prefill shapes.

The following table is chronological: "pending" in a row records the decision at that trial, and the final
qualification and shipped choice appear below it.

| Trial | Exact knob change from recipe | M=16 result | M=64 result | Decision at trial |
| --- | --- | --- | --- | --- |
| Baseline | Explicit source file, `WORK=` fallback; native `f1x2/k4`, `d2/smem-async` | 167.4 µs whole graph (8/4) | 399.9 µs whole graph (8/4) | Accepted audited comparison; same seven schedules as 12/5 baseline. |
| Wider first gate/up N fragment | Only M=64 `TILE@place_c0904cfc6e=mma_m16n8k64_e2m1_f32/f1x4/k4`, fixed `WORK=w1x1`, `STAGE=d2/smem-async` | Not tried | Compile refused: “its kernel pins … leave no schedule row this kernel offers.” | Reject this tuple; no timing or instruction claim. |
| Shallower first gate/up stage | Only M=64 `STAGE@place_c0904cfc6e=d1/smem-async`, fixed native `f1x2/k4`, `WORK=w1x1` | Not tried | Native MMA; first piece 165.1 µs vs baseline 159.5; whole graph 407.8 vs 399.9 µs (8/4). Shared memory 15→7.5 KiB, reported occupancy 12→27%. | Reject; more occupancy did not overcome less pipelining. |
| Wider second gate/up WORK | Only M=64 `WORK@place_8f9ed3f314=w1x2`, fixed native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; second piece 123.3 vs baseline 153.4 µs, whole graph 379.5 vs 399.9 µs (8/4). Grid 2176×32→1088×64, shared memory 25→45 KiB. | Provisional winner; repeat timing and independent MLP oracle pending. |
| Wider down WORK | Only M=64 `WORK@node_linear_2=w1x4`, fixed native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; down 63.2 vs baseline 76.4 µs, whole graph 390.7 vs 399.9 µs (8/4). Grid 640×64→320×128, shared memory 15→25 KiB. | Promising; test combined route and oracle. |
| Combined second gate/up and down WORK | M=64 `WORK@place_8f9ed3f314=w1x2`, `WORK@node_linear_2=w1x4`; all other exact pins unchanged | Not tried | Three native contraction kernels remain. Second gate/up 122.6 µs, down 62.3 µs, whole graph 361.3 µs (12/5), versus audited baseline 399.9 µs (8/4). | Provisional M=64 winner; matched repeat and stock MLP oracle pending. |
| Wider second gate/up WORK on decode | Only M=16 `WORK@place_4b5e95ec28=w1x2`, fixed native `f1x2/k4`, `d2/smem-async` | Compile refused: “its kernel pins … leave no schedule row this kernel offers.” | Not tried | Unclassified geometry restriction; no timing or instruction claim. |
| Wider down WORK on decode | Only M=16 `WORK@node_linear_2=w1x4`, fixed native `f1x2/k4`, `d2/smem-async` | Native MMA; down 23.7 vs baseline 30.0 µs, whole graph 165.7 vs 167.4 µs (8/4). Grid 160×64→80×128, shared memory 15→25 KiB. | Not tried | Keep as candidate; end-to-end gain is near run noise. |
| Deeper second gate/up stage on decode | Only M=16 `STAGE@place_4b5e95ec28=d3/smem-async`, fixed native `f1x2/k4`, `WORK=w1x1` | Native MMA; target kernel 61.0 µs vs baseline 61.0, whole graph 163.2 vs 167.4 µs (8/4). Shared memory 15→22.5 KiB. | Not tried | Reject; kernel did not improve, whole graph change is noise from other kernels. |
| Wider first gate/up M work | Only M=64 `WORK@place_c0904cfc6e=w2x1`, fixed native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; first piece 120.7 vs baseline 159.5 µs, whole graph 365.6 vs 399.9 µs (8/4). Grid 4352×32→2176×64, shared memory 15→20 KiB, reported occupancy 12→21%. | Provisional M=64 first-piece win; test combined route and oracle. |
| Four-warp M expansion | Only M=64 `WORK@place_c0904cfc6e=w4x1`, fixed native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; first piece 101.8 vs baseline 159.5 µs, whole graph 349.1 vs 399.9 µs (8/4). Grid 4352×32→1088×128, shared memory 15→30 KiB, occupancy 12→25%. | Better first-piece point; one w8x1 boundary trial pending. |
| Eight-warp M boundary | Only M=64 `WORK@place_c0904cfc6e=w8x1`, fixed native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA but first piece 622.4 µs and whole graph 865.3 µs (8/4). Grid stayed 1088 while block doubled 128→256 threads; shared memory 30→50 KiB, registers 66→48, reported local bytes 0. | Reject sharply; M=64 offers no further grid reduction beyond w4x1. Exact performance cause unestablished. |
| Three-piece WORK candidate | M=64 first gate/up `w4x1`, second gate/up `w1x2`, down `w1x4`; native `f1x2/k4`, `d2/smem-async` unchanged | Not tried | Pin audit clean; all three native MMA. Kernel medians 96.9/121.7/64.0 µs; whole graph 294.8 µs (12/5) vs baseline 399.9 µs (8/4), a provisional 26.3% isolated reduction. | Current M=64 candidate; matched repeat, independent MLP oracle, golden and endpoint checks pending. |
| TMA on four-warp first piece | M=64 first gate/up `WORK=w4x1`, `STAGE=d2/smem-tma/p2`; other pieces baseline | Not tried | Native MMA and actual TMA stage; target 84.2 vs 101.8 µs with `w4x1,d2/smem-async`, whole graph 336.1 vs 349.1 µs (8/4 single runs). Shared memory 30→27 KiB, same 1088×128 grid. | Promising transport choice; combine and repeat before promotion. |
| M-expanded second gate/up | Only M=64 `WORK@place_8f9ed3f314=w2x2`, native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; target 75.8 vs baseline 153.4 µs and prior `w1x2` 123.3 µs. Whole graph 333.2 vs baseline 399.9 µs (8/4); grid 2176×32→544×128, shared memory 25→50 KiB. | Stronger second-piece point; test w4x2 boundary and combined route. |
| Four-by-two second gate/up | Only M=64 `WORK@place_8f9ed3f314=w4x2`, native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; target 49.8 vs baseline 153.4 µs, whole graph 300.7 vs 399.9 µs (8/4). Grid 2176×32→272×256, shared memory 25→60 KiB, registers 106→64, local bytes 8. | Strongest second-piece point; validate combined route and oracle. |
| M-expanded down | Only M=64 `WORK@node_linear_2=w2x2`, native `f1x2/k4`, `d2/smem-async` | Not tried | Native MMA; down 61.2 vs baseline 76.4 µs, whole graph 388.8 vs 399.9 µs (8/4). Grid 640×64→320×128, shared memory 15→20 KiB. | Similar to `w1x4`; use in the combined candidate. |
| First-piece async `/p2` control | M=64 first gate/up `WORK=w4x1`, `STAGE=d2/smem-async/p2`, native `f1x2/k4`; others baseline | Not tried | Native MMA; target 99.3 µs, whole graph 345.2 µs (12/5). Same geometry as the TMA `/p2` run, whose target was 84.2 µs. | `/p2` alone did not account for the TMA gain. |
| Three-piece async finalist | M=64 first gate/up `w4x1`, second `w4x2`, down `w2x2`; native `f1x2/k4`, `d2/smem-async` | Not tried | All three native MMA; kernel medians 100.0/51.3/60.8 µs; whole graph 221.7 µs (12/5) versus audited baseline 399.9 µs (8/4). | Independent stock oracle passed at 64 and 37 active rows; repeat matched timings and golden replay next. |
| Three-piece TMA `/p2` finalist | Same WORK as async finalist; only first gate/up `STAGE=d2/smem-tma/p2` | Not tried | Isolated graph 205.4 µs (12/5), native kernels 84.2/49.2/60.5 µs. However first full-size MLP `run_device` failed `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` during whole-program graph capture. | Not serving-qualified. A later generic first-capture fix passed its GPU regression; async remains the shipped route. |

The sweep stopped weak or refused families instead of repairing the general prior. Rejections are recorded
with the exact piece scope, pin tuple, compiler message, emitted instruction result, and observed time if runnable.
No candidate gets a serving claim from isolated timings alone.

The M=64 `f1x4/k4` refusal above is a **likely layout restriction**, not an established compiler bug. The saved
post-cut first gate/up child produces BF16 `[64,1088,16]` values with the innermost N block axis of extent 16;
`_block_scaled_warp_stage` declines an N tile with a mask. An `f1x4` fragment plausibly requires N32 and would
mask that inner block. This explanation is a source-backed inference; the exact rejected tile mask has not yet been
printed. The usable `f1x2/k4` route is the workaround. The M=16 WORK refusal remains unclassified.

The TMA refusal is a confirmed serving-path compatibility failure, not a numerical comparison: the isolated
`emmy run --ir --bench` path completed, while `compare_stock_vllm_nvfp4_mlp.py --static-only --static-rows 64
--rows 64` with the same scoped `d2/smem-tma/p2` first-piece pin failed on its first `run_device` graph capture.
`CompiledProgram.capture_program_graph` reached the runtime's descriptor setup inside CUDA capture, which attempts
stream synchronization. The per-launch capture path initializes descriptors before capture. A bounded generic runtime fix now calls `ensure_descriptors` before first whole-program capture, matching the
per-launch capture path. The focused first-capture TMA GPU regression passes on the RTX 5090 after rebuilding
the actual Rust extension with `setuptools-rust` and verifying the `.so` was replaced. The first attempted
`pip install --no-build-isolation` without that build backend produced a pure-Python wheel and did **not**
validate the patch. The full-size TMA candidate has not been requalified for serving; `d2/smem-async` remains
the measured default and avoids this transport path.

The first golden-recording attempt also rejected a correct FP4 intermediate comparison. For
`mul_static_fp4_bits` at `[64,8704]`, CUDA produced packed byte `0x88` where eager produced `0x00`:
both decode to zero, with only the signs of the zero nibbles different. The old correctness path
compared these carriers as unsigned integers and reported a wrong answer. The narrow fix decodes
both sides of an `f4e2m1x2` output before comparison; ordinary `u8` outputs retain their integer semantics and are not decoded.
A focused test checks signed-zero acceptance, nonzero-code rejection, and unchanged `u8` behavior.
The initial “reference self-disagreement” wording was misleading for this run: the saved reference
was eager output from the same input, not another execution of the greedy kernel. The six realization routes were re-recorded after the fix. Canonical golden check and strict evaluation now pass
for both M=16 and M=64 with an empty tune DB and no schedule or placement environment pins.

An additional CLI comparison gap remains outside this bounded fix: `_wrong_answer_flag` returns no error for
`actual=[NaN], reference=[1.]` and `actual=[1.], reference=[inf]`, because `max(0, NaN)` retains zero. This is a
function-level CPU reproduction, not observed serving corruption. Candidate admission used separate finite-output
checks and the exact-checkpoint stock MLP oracle. The generic comparator should reject mismatched nonfinite
values while allowing matching masks in a follow-up.

The synthetic `--ir` source was also a weak numerical witness. Its random FP4 weight source came
from `standard_normal * 0.02`, then the packed storage path cast those small floats to `uint8`;
the observed synthetic checkpoint codes were all zero. A replay using that source therefore cannot
establish behavior on nonzero checkpoint weights. The separate stock vLLM comparison below used
the exact quantized checkpoint and remains independent evidence for the async finalist.

The exact-checkpoint vLLM 0.23 stock quantized MLP oracle completed on the async finalist. Layer 0 BF16 output
relative RMS was 0.5469% at all 64 active rows and 0.5444% at 37 active rows. The M=16 baseline decode bucket
with one active row produced 0.5470% on its separately sampled input and remained independent of stale padded
rows. All three had zero elements outside the existing provisional `atol=rtol=0.05` diagnostic. These are
same-checkpoint MLP comparisons, not a full-model quality equivalence claim. The M=16 sample included a padding
probe that changes its RNG position; do not compare its RMS directly to older runs as identical-input evidence.

For one deterministic M=64 input, the audited baseline and three-piece async finalist returned bit-identical
BF16 output bytes (`sha256:3a882842a714339d6029fdc529bc324d043b2b0fcc4909969774fa38a0a67fb8`).
The finalist therefore changes schedule without adding numerical drift on that sampled input. A hand-authored
A–B–B–A warm repeat of the existing `emmy run --ir --bench --bench-backends emmy --warmup 5 --iters 20`
command measured baseline whole-program minima 410.8 and 410.4 µs and async finalist 226.8 and 228.6 µs.
Their mean minima differ by 44.5%. These are **synthetic IR microbenchmarks**: the source generator produced
zero packed FP4 weight codes, so this reduction is a shape/graph diagnostic, not a measured full-model speedup.

The six kernel realizations were recorded from the exact pinned M=16/M=64 graph into the model-specific
`recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json`. Standalone activation-scale roots initially recorded
`PLACE=fuse`, which produced a slow scalar grid-one realization (2,104.5 µs M=64). This was an inappropriate
route selection, not a numerical compiler bug. Explicit cuts at `PLACE@map.1/map=cut` and
`PLACE@map.2/map=cut` recreate the exact two output-piece identities of the full MLP for both shapes;
their M=64 end-to-end realization is 3.9 µs and M=16 is 4.1 µs. The canonical golden holds 20 kernel
fingerprints, 16 measured schedule rows, and six measured routing decisions. On the 5090, run:

```sh
emmy golden check recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json
EMMY_TUNE_DB=/tmp/qwen38-golden-only-empty.db \
  emmy eval golden --golden recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json \
  --serving-config recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env
```

`golden check` reports current; strict evaluation reports two twins deploy from golden rows alone and zero
failures, with no schedule or placement pins in the environment. The sole serving launcher uses this artifact,
`--strict-evidence`, and a fresh private tune DB per boot. A preexisting tune DB can contain other measured
rows that strict evidence legitimately permits, so the launcher isolates the DB to make golden-only replay
reproducible. The first full boot imported exactly these 16 rows, compiled both programs, and reached health
200 on host port 8080. The final golden-only server compiled all 64 layers, used seven kernels per shape with
three native FP4 contraction kernels, and admitted no inherited hand pins or preexisting tune DB evidence.
The live M=64 log shortens the first cut child's symbol to `...place_f688369f74`, whereas the recorded row is
`...place_f688369f74__place_c0904cfc6e`. The live symbol's only cached cubin was created during the first
golden-only boot and has 66 registers and 31,744 bytes of static shared memory, matching the measured `w4x1`
first-piece resource (about 30 KiB, 66 registers). The older `...place_c0904cfc6e` cubin predates golden-only
boot and has 74 registers/16,384 bytes, matching the original `w1x1` resource. The second gate/up and down
live names match the `w4x2` and `w2x2` golden rows. This is a strict-row and cached-resource inference, not a
direct dump of the running executor's schedule objects. It resolves the apparent name mismatch without claiming
a measured end-to-end tuning gain.

## Paired serving measurements and qualification

The checked-in [fixed prompts](nvfp4-qwen-mlp-tuning-prompts.jsonl) are coherent engineering-report summary and
numeric extraction tasks. They were formatted with the pinned tokenizer's chat template and thinking disabled,
then sent to `/v1/completions`. Actual prompt lengths include template tokens. For the official vLLM 0.23
`CustomDataset` client, use `--custom-output-len -1` so each fixture's output budget is respected; its default
256 would override the fixture. The stock image lacked `pandas`, so the realistic stock and final golden runs
used the same CPU-only dev-image benchmark client over the host port. Original baseline and first golden runs
used that dev image as an in-container client; cross-run and client-topology variation limits fine-grained
baseline-versus-finalist comparisons. All servers used the same BF16 checkpoint, eager/no outer CUDA graphs,
one request, cap 64, max context 4096, temperature zero, and the same prompt fixture. The 5/16 benchmark used
`--ignore-eos`, one warmup and five measured requests. Natural-EOS realistic tasks used one warmup and three
measured requests; output counts differ, so total wall times are descriptive rather than output-matched ratios.
To reproduce the realistic client run, render each fixture's `prompt` with the pinned tokenizer's
`apply_chat_template([{"role":"user","content":prompt}], tokenize=False,
add_generation_prompt=True, enable_thinking=False)` and preserve its `output_tokens` field. Repeat one rendered
row four times into a JSONL file for each task, then use the vLLM 0.23 dev-image client (with `pandas`) against
each server in turn:

```sh
vllm bench serve --backend openai --base-url http://127.0.0.1:8080 \
  --endpoint /v1/completions --model Inferact/Qwen3.8-27B-NVFP4 \
  --tokenizer Inferact/Qwen3.8-27B-NVFP4 --dataset-name custom \
  --dataset-path /tmp/qwen38-one-rendered-task.jsonl --custom-output-len -1 \
  --num-warmups 1 --num-prompts 3 --max-concurrency 1 --request-rate inf \
  --temperature 0 --seed 42 --save-result --save-detailed
```

The dev-image client used host networking; the server used host port 8080. For the controlled short benchmark,
substitute `--dataset-name random --random-input-len 5 --random-output-len 16 --random-prefix-len 0`,
`--num-prompts 5`, and `--ignore-eos`. The saved benchmark JSON records the actual input/output lengths.
The [raw standard client results](evidence/qwen38_nvfp4_5090) retain per-request TTFTs, inter-token intervals,
generated text, output lengths, failures, and aggregate fields for the nine table rows, the three natural-EOS
quality checks, and the near-4K shape check. They contain no synthetic reinterpretation of client timing.

| Workload / arm | Input→output tokens | TTFT ms | TPOT ms (tok/s) | Total wall / request |
| --- | ---: | ---: | ---: | ---: |
| Short 5/16, stock vLLM matched eager | 5→16 | 305.33 | 116.89 (8.55) | 2.06 s |
| Short 5/16, original pinned mixed | 5→16 | 257.24 | 95.04 (10.52) | 1.68 s |
| Short 5/16, final golden repeat | 5→16 | 261.91 | 95.76 (10.44) | 1.70 s |
| Summary, stock vLLM matched eager | 818→58 | 2223.13 | 118.49 (8.44) | 8.98 s |
| Summary, original pinned mixed | 818→65 | 2006.01 | 103.70 (9.64) | 8.64 s |
| Summary, final golden repeat | 818→65 | 1997.50 | 102.89 (9.72) | 8.58 s |
| Extraction, stock vLLM matched eager | 3082→128 | 8147.25 | 121.50 (8.23) | 23.58 s |
| Extraction, original pinned mixed | 3082→128 | 7200.50 | 101.46 (9.86) | 20.09 s |
| Extraction, final golden repeat | 3082→128 | 7305.81 | 102.64 (9.74) | 20.34 s |

The first golden boot, before the native runtime rebuild and using the in-container client, measured short
265.41 ms TTFT/97.42 ms TPOT, summary 2030.45/104.22, and extraction 7336.70/103.80. A separate stock
short in-container run measured 293.93/112.15, versus 305.33/116.89 from the external client above. These
are observed cross-run/client-path differences; their cause is unisolated. The repeat after rebuilding the
runtime used the same external client as stock. The hand-tuned M=64 golden has a large synthetic isolated
MLP timing win, but **no clear end-to-end improvement over the original pinned mixed route** in these small
samples; the long extraction repeat is slightly slower. Both mixed routes outperform matched eager stock in
these measurements, which does not imply they beat other stock deployment configurations.
The next useful performance measurement is actual-checkpoint MLP time inside serving (or a replay with
representative packed weights and working set) to locate where the isolated synthetic gain disappears before
expanding another knob sweep; the current results do not establish its cause.

The summary responses from all arms satisfied the four-bullet numeric summary request. For extraction,
the 128-token budget clipped every arm; a separate natural-EOS max-256 run ended at stock 131 tokens and
mixed 136 tokens. Both mixed answers correctly extracted the requested post-merge 277 ms and 9.59 tok/s and
two caveats. Stock extracted the numbers but its second caveat was inaccurate against the supplied text.
This single illustrative task is not broad quality evidence. Earlier five fixed correctness tasks, including
a 3,530-token needle, passed; selected-token logprobs still differ from stock. The exact-checkpoint MLP oracle
above bounds numerical error on sampled inputs, not whole-model equivalence.

The final golden server also completed a 4,005-input/16-output near-4K request (TTFT 9303.57 ms, TPOT
95.36 ms, 10.73 s total, zero failures). A chat stream was cancelled **after actual text `1`**, then a fresh
deterministic request answered `Tokyo` and `/health` returned HTTP 200. The server remains reachable on host
port 8080. Remaining release work is `make test-priors`, final CI, and review of the
documented comparator/TMA limitations; no further GPU tuning is required for this bounded round.

Adding a repository model golden requires refitting both offline priors even though this serving launcher uses
strict measured evidence. The repository import included this golden's 16 performance rows and six routes.
Default schedule export completed successfully after roughly 90–93 minutes of active CPU enumeration, producing
750 golden and 828 measured groups; 61 golden rows were skipped by the exporter. Two logged producer-identity
skips were in older Qwen3.8 V100 golden families, outside the new NVFP4 file; no causal compiler regression is
established from those logs. Placement and schedule refits both wrote new weight artifacts. The required
`make test-priors` schedule reproduction nodes are still running; their result belongs in the final gate report.

## Automated review disposition

The review claim that symbolic prefill is fixed at eight rows is incorrect: eight rows is a trace example.
`MLPPrograms` gives axis 0 the symbolic `num_tokens` with capacity 64, and `run_device_sym` binds the actual row
count and returns the actual prefix. The deployed vLLM adapter currently chooses the static M=64 bucket; the
symbolic path remains available for future integration. A strengthened CPU trace test checks the symbolic axis.
No >8-row symbolic GPU execution is claimed by that test.

The later review claim that removing stock MLP modules makes the pinned vLLM weight load silently lose MLP
tensors is also incorrect for this startup path. `EmmyQwen35MlpModel.load_weights` consumes the lazy vLLM
iterator, intercepts the exact 12 expected checkpoint leaves per layer (three projections times four leaves),
rejects duplicate, missing, and unclaimed MLP keys, and passes every remaining tensor to the stock Qwen
loader. `MLPPrograms` then binds those packed tensors independently from the same local checkpoint.
The actual 64-layer pinned engine boot and deterministic requests passed with this loader. MLP parameters are
intentionally absent from the replacement modules' `named_parameters`; a future hot weight-sync path is not
qualified by the startup test and would need its own explicit ownership contract.
