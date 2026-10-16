# OLMoE-1B-7B-0125-Instruct with one layer's experts in host memory on one RTX 5090

## Result

Reading a layer's routed experts in place from mapped host memory works with no compiler change and changes no token,
and the expert kernels reach 85% of the copy engine's rate on this link. With layer 8's 64 experts in pinned host
memory (`EMMY_GEN_EXPERTS_HOST=8`), all 8 greedy chat answers came back token for token the same as with every expert
on the GPU (512 tokens). Decode at one stream went from 2.94 ms to 35.81 ms per token. The added 32.9 ms is that
layer's 8 hit experts per token, 12 MiB each, crossing the link: 100.7 MB at the 3.59 GB/s a copy-engine copy from
pinned memory measures on this box is 28.0 ms, so one launch reading its expert in place costs 4.1 ms against a 3.5 ms
floor.

| Experts in host memory | Median TPOT | Median ITL | p99 ITL | Median TTFT (256 tokens) | Tokens equal to all-GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 2.94 ms | 2.87 ms | 3.72 ms | 38.9 ms | (reference) |
| layer 8 | 35.81 ms | 35.75 ms | 36.53 ms | 225.2 ms | 8 of 8 answers |

The link is the whole story here, and it is this box's: the 5090 sits in a slot that trains to **PCIe 5.0 x1**, so
3.59 GB/s is all a copy gets. A x16 slot gives about 15 times that, which would put the same layer near 2 ms per token,
still about 11 times the whole layer's GPU-resident cost. Strategy A (every cold expert read in place) is therefore a
floor, not a plan: a host layer has to be hit rarely, or its experts copied ahead, to be affordable.

For the plan's question 3 (copy or read in place), reads in place from the expert kernels come within 15% of a
copy-engine copy of the same bytes, so a copy into a GPU cache only pays when the expert is used again soon. The
gap is small enough that the in-kernel fetch (strategy B) and an SM copy kernel (strategy D) gain little over A on a
single use; M1 on a x16 card, where latency rather than the link may bind, still has to confirm it.

## Which schedules read host memory

Decode ran the fixed-slot tier: each of the 8 slots launches the M=1 expert program with its weights fetched through
the indirect table. The golden's evidence picked a tensor-core gate/up kernel staged with cp.async
(`STAGE=d2/smem-async`) and a down projection with plain loads and a cooperative reduce (`WORK=t128`, `REDUCE=coop`).
Both read mapped host memory correctly. No TMA schedule was in play, so the plan's expected trap (TMA cannot take an
indirect operand) did not arise.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/zero_copy_experts_rtx5090 --local` served
`allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's vLLM
0.23 plugin (the generate runner), from the image built at commit `9a4a0934e` (image id `sha256:342ffc5f59fa…`, in
`image.json`), with the serving shape of the recipe's pinned config and `EMMY_STRICT_EVIDENCE=1`. Each row sent the
first 8 prompts of `HuggingFaceH4/ultrachat_200k` greedily, 64 output tokens each, with token ids returned, then ran
`vllm bench serve` at concurrency 1 on 4 random 256-token prompts with 64 output tokens, `--ignore-eos`. The two rows
are two server boots of the same image; only `EMMY_GEN_EXPERTS_HOST` differs.

The link rate is a 256 MiB `cudaMemcpyAsync` from pinned memory, timed over five copies on the same card outside the
server: 3.59 GB/s. `nvidia-smi` reports the link at generation 5, width 1 of 16.

## Limitations

- One layer, decode at one stream, one card. The all-16-layer row was dropped: pinning 12 GiB of experts beside a
  desktop session took the box to 1.7 GB of available memory, and the engine died during boot.
- A x1 link. The per-launch overhead over the floor (0.6 ms of 4.1 ms) may not stay 15% on a x16 link, where each
  expert's 12 MiB takes about 0.2 ms; the 5090 in a x16 slot, or the H100, has to measure it.
- No nsys: the bytes over PCIe are inferred from the expert size and the routing, not counted.

## Run and platform

Run `20261016T060644Z`, started `2026-10-16T06:06:44Z`, two rows, both succeeded. One RTX 5090 (32 GB, driver
580.178.04, PCIe 5.0 x1) on Ubuntu 24.04 with kernel 7.0.0, shared with a desktop session that held about 5 GB of the
card; the server used a 0.75 memory fraction.

Archive: `results_rtx5090x1.tar.gz` (Git LFS), root `2026-10-16_06-06-44/`, holding the experiment record, the
benchmark logs (which also show the dropped third row's failure), and per-row `bench_c1.json`, `image.json`,
`responses.jsonl` and `server.log`. Node addresses in the raw files read `node`.
