# Neptune schedule replay on A100 40GB

## Conclusion

The five paper-table operator families reproduce on the same NVIDIA A100-SXM4-40GB product used by the Neptune
paper. The published library-relative ratios and this replay differ by 1.3--5.1% across the five families. The
directions are unchanged: optimized libraries lead on the three prefill families, causal decode is near parity, and
Neptune leads on GQA decode.

This run did not tune Neptune. It replayed the fixed schedules and the saved schedules from the prior A100 80GB
search through Neptune's stock evaluator. The saved schedules therefore establish portability to the 40GB card, not
optimality on it.

The current-PyTorch comparison also reproduces. Neptune is 2.98x faster than PyTorch 2.13 full-graph
`torch.compile` on GQA decode by geometric mean and wins all eight shapes in that family. Neptune wins 15 of all 40
shapes. It is slower on global, causal, and GQA prefill by geometric-mean factors of 1.20x, 1.14x, and 1.15x,
respectively, and is approximately tied on causal decode at 0.99x the Inductor latency.

The move from the earlier A100 80GB run slows Neptune and the PyTorch 2.13 comparator almost identically. Across the
40 matched shapes, Neptune latency is 1.238x higher and Inductor latency is 1.240x higher by geometric mean. The
stable cross-system ratios are therefore not caused by Neptune alone becoming slower.

## Alignment with the Neptune paper

The [Neptune paper](https://arxiv.org/abs/2510.08726) reports each implementation relative to the fastest manually
optimized library in Table 4. The replay follows the public plotting code's library set: PyTorch FlashAttention,
PyTorch memory-efficient attention, cuDNN, Tri Dao FlashAttention, and FlashInfer. For each shape, it takes the
arithmetic mean of 15 projected GPU ranges for each implementation, selects the fastest Neptune schedule and the
fastest library, and then takes the geometric mean over eight sequence lengths.

| Operator | Published library / Neptune | Replayed library / Neptune | Difference |
| --- | ---: | ---: | ---: |
| Prefill global | 0.84x | 0.80x | -5.1% |
| Prefill causal | 0.81x | 0.78x | -4.0% |
| Prefill GQA | 0.80x | 0.79x | -1.3% |
| Decode causal | 0.99x | 1.03x | +3.7% |
| Decode GQA | 1.24x | 1.21x | -2.4% |

The [public repository](https://github.com/uiuc-arc/neptune) and
[Zenodo artifact](https://zenodo.org/records/19293880) provide source, evaluator, and plotting code but no raw A100
profiles. The repository plotting code reads ignored `logs/results_a100/profiles` files. The raw profiles retained by
this replay are therefore the evidence for the reproduced column.

## Current PyTorch comparison

The PyTorch lane uses two fresh processes for each shape. Each process runs one warmup followed by 15 captured
CUDA-event measurements and records the minimum. The per-shape comparator is the arithmetic mean of the two minima.
All 80 compiled measurements passed eager correctness at `rtol=1e-3, atol=1e-3`, used `fullgraph=True`, and captured
the complete forward pass in a CUDA Graph.

| Operator | Replayed Neptune / Inductor | Paper Neptune / current Inductor | Neptune wins |
| --- | ---: | ---: | ---: |
| Prefill global | 1.20x | 1.14x | 1/8 |
| Prefill causal | 1.14x | 1.09x | 2/8 |
| Prefill GQA | 1.15x | 1.14x | 1/8 |
| Decode causal | 0.99x | 1.02x | 3/8 |
| Decode GQA | 0.34x | 0.33x | 8/8 |

Lower values favor Neptune in this table. The paper column uses the reproduced library as a bridge:

```text
paper Neptune / current Inductor
  = replayed Neptune / current Inductor
  * replayed library / Neptune
  / published library / Neptune
```

At GQA-decode sequence lengths 2048 and 32768, the representative Neptune and Inductor latencies are 19.8 / 57.3
microseconds and 143.2 / 615.9 microseconds, respectively. The Neptune value is the mean of 15 projected GPU ranges;
the Inductor value is the mean of two independent minimum-of-15 measurements.

## Protocol and limitations

- Neptune ran revision `3aa55c12ac822337e630b809b0d9eabb11eee5d3` in image
  `evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`.
- The artifact environment uses PyTorch 2.6.0 and CUDA 12.4. The current comparator uses PyTorch 2.13.0 and CUDA
  13.0. The ratios are kernel-level evidence from separate processes, not an end-to-end application result.
- The machine has one NVIDIA A100-SXM4-40GB with 40960 MiB, NVIDIA driver 580.173.02, and a 400 W power limit.
- Two stock-evaluator experiment records are marked failed because SSH disconnected while returning large archives.
  All 40 profiles had completed on the host, all 40 converted successfully, and their raw artifacts are retained.
  The final GQA shell status is 141 because the output pipe closed after the last profile completed.
- The PyTorch replay succeeded for all five experiment rows and all 80 measurements. Setup-only failed attempts are
  not included in the result archive.

## Durable files

- `paper-baselines.csv` contains all 40 shapes, both PyTorch repetitions, Neptune and library means and minima, and
  the ratios used above.
- `paper-table.csv` contains the five displayed paper rows.
- `results_a10040.tar.gz` contains the two final run directories, ten experiment records, five stock-evaluator
  artifacts with the 40 raw Nsight profiles, five PyTorch artifacts, 80 current-PyTorch JSON measurements, and 40
  `nsys stats` CSV exports. Its SHA-256 is
  `7f8756af3b02bc434c5b9768f4a78214298840b12a9319cddbddc95cc9c9d356`.
- The stock evaluator started at `2026-09-08T20:41:33Z`; the successful PyTorch comparator started at
  `2026-09-08T22:55:35Z`.
