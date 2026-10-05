# Llama 3.1 8B Instruct FP16 on one 32 GB V100

Status: serving-qualified through a 59,999-token prompt on one Tesla V100-SXM3-32GB with the pinned, pullable
1Cat/vLLM image. The Emmy compiler golden is complete on this GPU; Emmy serving remains unqualified.

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

Measured 2026-10-04 and retuned 2026-10-05 on a rented Tesla V100-SXM3-32GB, SM70, with deployable `-O3`
compilation, five warmups and 20 timed iterations per target. The golden holds nine traced programs: six
pre/post-attention serving twins at symbolic, M1 and M8 widths, representing all 32 identical decoder layers, plus
embeddings, final normalization and the output head. vLLM owns attention and LoRA application. The file has 44
fresh-lowered kernels, nine measured program targets, 39 measured kernel rows and seven routing decisions. Every target
passed strict correctness and has positive Emmy, eager PyTorch and `torch.compile` timings; every kernel row has a
positive same-input greedy reference.

| Target | Emmy (µs) | Eager (µs) | `torch.compile` (µs) |
| --- | ---: | ---: | ---: |
| Post-attention, symbolic | 3,857 | 2,621 | 2,322 |
| Post-attention, M1 | 627 | 637 | 423 |
| Post-attention, M8 | 1,212 | 719 | 662 |
| Pre-attention, symbolic | 1,125 | 592 | 360 |
| Pre-attention, M1 | 194 | 127 | 64 |
| Pre-attention, M8 | 340 | 151 | 93 |
| Final normalization, M1 | 2.11 | 41.31 | 5.15 |
| Embeddings, M1 | 1.37 | 6.10 | 4.05 |
| Output head, M1 | 1,758 | 1,254 | 1,097 |

Cuts and smaller matrix tiles removed the worst duplicated work and register spills. Two further measured matrix
schedules with `f2x2/k8` tiles reduced the post-attention M1 path from 1,491 to 627 µs (2.38×) on 2026-10-05;
the full path passed strict eager correctness and now matches eager latency. Four cooperative reductions replaced
slower serial reductions, improving their four full targets by 1.07–1.55×. The six layer paths and output head still
miss the `torch.compile` speed bar by 1.5–3.7×; these are schedule and code-generation losses, not missing
coverage. The symbolic post-attention path is the largest remaining cost. Final normalization and embeddings beat
`torch.compile` by 2.4× and 3.0× respectively. `emmy golden check` confirmed all nine programs match fresh lowering;
`emmy eval golden` confirmed that all six serving twins compile from this golden's evidence alone on the exact V100.
No Emmy kernel was used in the qualified stock-vLLM serving measurement above.

## Limits

- The material prompt tested 59,999 tokens; 65,536 is the configured maximum, not a measured prompt length.
- The test LoRA adapter is selectable through the companion stock-vLLM recipe. Emmy's compiled serving plugin does
  not apply per-request LoRA updates in this qualification.
- Requalify after changing the image, model or tokenizer revision, driver, context length, or attention backend.
