# Llama 3.1 8B Instruct FP16 on one 32 GB V100

Status: serving-qualified through a 59,999-token prompt on one Tesla V100-SXM3-32GB with the pinned, pullable
1Cat/vLLM image. Emmy compiler serving is not qualified for this model on V100.

## Qualified deployment

| Item | Value |
| --- | --- |
| Hardware | 1× Tesla V100-SXM3-32GB, SM70 |
| Context / maximum sequences | 65,536 tokens / 8 |
| Tensor parallelism / dtype | 1 / FP16 |
| Driver / host CUDA | 580.178.04 / 12.9.86 |
| Model revision | `d10aef7999a2b5ba950ab3974312feeedbfe0b77` |
| Image digest | `sha256:6f34e0b247a78ca65f88f305b1f1cc52c9020ecb83a5ca21df0599676dc443d3` |
| KV capacity reported at boot | 89,104 tokens |

The official Meta repository is gated, so the recipe pins the public NousResearch mirror. Its `config.json` and all
four safetensor shard SHA-256 values were checked against Meta revision
`0e9e39f249a16976918f6564b8830bc894c89659` and are identical. The model has 8,030,261,248 parameters and
occupies about 15 GiB in FP16. The mirror omits Meta's tool-aware tokenizer template, so the recipe pins the tokenizer
from the qualified AWQ repository; its tokenizer config is byte-identical to Meta's.

The native 131,072-token context failed the earlier KV fit gate. The 65,536-token configuration loaded with a reported
1.36× concurrency at that full length. `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` selects the paged-KV gather path:
without it, 39/40 requests failed in each concurrency-8 base and adapter row with a 114,176-byte shared-memory
request against V100's 96 KiB limit. Prefix caching is disabled for this Volta path.

## Serving performance

Measured 2026-10-03 with three repeats of 32 requests, 512 input tokens, 256 forced output tokens, concurrency 8,
greedy decoding, and two warmup requests. All 96 measured requests succeeded.

| Output tok/s | Requests/s | Mean TTFT | Mean TPOT |
| ---: | ---: | ---: | ---: |
| 237.01 ± 0.63 | 0.93 | 1201.5 ms | 29.16 ms |

The output-throughput repeats were 236.28, 237.35, and 237.40 tokens/s. Model load and warmup took 87.87 seconds.
The raw experiment, system record, and separate tool and long-context probes are in
[`experiments/Meta-Llama-3.1-8B-Instruct/serving_v100_sxm3_32gb`](../../experiments/Meta-Llama-3.1-8B-Instruct/serving_v100_sxm3_32gb/RESULTS.md).
The server returned a parsed `get_weather` call for Paris. A 59,999-input-token request generated three tokens
without OOM; its time to first token was 113.00 seconds.

The 2026-08-13 run reported 280.80 ± 9.64 output tokens/s on a locally resolved image digest that is not pullable
from the registry. The current result is 15.6% lower, but the image, driver, and attention path all changed, so this
is a historical comparison rather than an isolated regression measurement. The reduced-memory 0.65 recipe variant
remains unverified.

## Emmy compiler qualification

The current compiler traced six distinct serving-twin targets on the exact V100. All six pass the fresh-lowering check.
A short diagnostic with one warmup and one timed iteration produced these results; it does not meet the recording bar:

| Target | Emmy | Eager | Result |
| --- | ---: | ---: | --- |
| Post-attention, symbolic | — | — | Kernel exceeded 60 s |
| Post-attention, M1 | 1,193,433 µs | 757 µs | Correct |
| Post-attention, M8 | — | — | Kernel exceeded 2 s |
| Pre-attention, symbolic | 579,149 µs | 678 µs | Correct |
| Pre-attention, M1 | 390 µs | 138 µs | Correct |
| Pre-attention, M8 | 418 µs | 163 µs | Correct |

The `post1` target also passed a direct comparison with eager PyTorch at `rtol=atol=1e-3` (maximum absolute error
0.000488) in a longer run. Its greedy path took 1,184,860 µs versus 673 µs eager and 436 µs `torch.compile`. The
isolated re-benchmark exceeded the ten-second GPU-time limit and recorded `bench_fail`; a three-cut candidate did not
complete within a three-minute diagnostic limit. The trace does not cover embeddings, final normalization, and the
output head, so it is a partial compiler inventory. No complete, measured model golden was committed, and no Emmy
kernel was used for serving.

The earlier compiler run on 2026-08-13 measured 223,882 µs for a sequence-length-1 layer versus 773 µs eager, and
its 512-token prefill CUDA did not compile. The current compiler path is different, so those timings are historical.
The next tuning step is to find a cut and schedule that avoids duplicated matmul work, then record every target on
this exact GPU and finish the non-layer coverage. Loop fusion remains maximal; kernel boundaries are chosen later
from measured evidence.

## Limits

- The material prompt tested 59,999 tokens; 65,536 is the configured maximum, not a measured prompt length.
- The test LoRA adapter is selectable through the companion stock-vLLM recipe. Emmy's compiled serving plugin does
  not apply per-request LoRA updates in this qualification.
- Requalify after changing the image, model or tokenizer revision, driver, context length, or attention backend.
