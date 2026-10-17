# OLMoE-1B-7B-0125-Instruct with one layer's experts in host memory on the RTX 5090

## Result

Reading a layer's routed experts in place from mapped host memory works with no compiler change, changes no token,
and runs at the copy engine's rate. With layer 8's 64 experts in pinned host memory (`EMMY_GEN_EXPERTS_HOST=8`) on an
RTX 5090 with a PCIe 5.0 x16 link, decode at one stream went from 2.84 ms to 4.99 ms per token. The added 2.15 ms is
that layer's 8 hit experts per token, 12 MiB each, crossing the link: 100.7 MB at the 45.8 GB/s a copy-engine copy
from pinned memory measures on that machine is 2.20 ms. So one launch reading its expert in place costs what copying
the expert would, about 0.27 ms, and the greedy answers came back token for token the same as with every expert on the
GPU, on both machines.

| Machine | Link (measured copy rate) | Experts in host memory | Median TPOT | p99 ITL | Median TTFT (256 tokens) | Answers equal to all-GPU |
| --- | --- | --- | ---: | ---: | ---: | --- |
| x16 VM | 5.0 x16 (45.8 GB/s) | none | 2.84 ms | 2.99 ms | 48.6 ms | (reference) |
| x16 VM | 5.0 x16 (45.8 GB/s) | layer 8 | 4.99 ms | 5.15 ms | 59.5 ms | 8 of 8 |
| dev box | 5.0 x1 (3.59 GB/s) | none | 2.94 ms | 3.72 ms | 38.9 ms | (reference) |
| dev box | 5.0 x1 (3.59 GB/s) | layer 8 | 35.81 ms | 36.53 ms | 225.2 ms | 8 of 8 |

The dev box's 5090 trains to one lane, so there the same layer adds 32.9 ms against a 28.0 ms floor (85% of the copy
rate). The x16 row is the one to plan with; the x1 row shows the cost scales with the link and nothing else.

Even at x16, one host layer costs about 12 times that layer's GPU-resident share of TPOT (2.15 ms against 2.84 ms for
all 16 layers). Strategy A (every cold expert read in place) is the floor the plan expected, not a plan: a host expert
has to be hit rarely, or copied ahead, to be affordable.

For the plan's question 3 (copy or read in place), reads in place match a copy-engine copy of the same bytes. A copy
into a GPU cache before use (strategies B and D) therefore buys nothing on a single use; it pays only when the copied
expert is hit again soon, or when the copy overlaps other work (strategy E).

## Which schedules read host memory

Decode ran the fixed-slot tier: each of the 8 slots launches the M=1 expert program with its weights fetched through
the indirect table. The golden's evidence picked a tensor-core gate/up kernel staged with cp.async
(`STAGE=d2/smem-async`) and a down projection with plain loads and a cooperative reduce (`WORK=t128`, `REDUCE=coop`).
Both read mapped host memory correctly and at link rate. No TMA schedule was in play, so the plan's expected trap (TMA
cannot take an indirect operand) did not arise.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/zero_copy_experts_rtx5090` served
`allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's vLLM
0.23 plugin (the generate runner), from the image built at commit `9a4a0934e` on each machine, with the serving shape of
the recipe's pinned config and `EMMY_STRICT_EVIDENCE=1`. Each row sent the first 8 prompts of
`HuggingFaceH4/ultrachat_200k` greedily, 64 output tokens each, with token ids returned, then ran `vllm bench serve` at
concurrency 1 on 4 random 256-token prompts with 64 output tokens, `--ignore-eos`. The two rows on a machine are two
server boots of the same image; only `EMMY_GEN_EXPERTS_HOST` differs. The x16 run used `--ssh`, the dev box `--local`.

The link rate is a 256 MiB `cudaMemcpyAsync` from pinned memory, timed over five copies on the same card outside the
server.

## Limitations

- One layer, decode at one stream. The dev box's all-16-layer row was dropped: pinning 12 GiB of experts beside a
  desktop session took it to 1.7 GB of available memory, and the engine died during boot. The x16 VM has 23 GB of RAM,
  too little for it as well.
- The x16 VM needed setup the recipe does not carry: it had no outbound internet, so the model was copied in and the
  prompt download went through an SSH-tunneled proxy, and a 32 GB swap file let the boot load the weights in 23 GB of
  RAM. Neither touches the measured decode path; the host experts sit in pinned memory, which does not swap.
- No nsys: the bytes over PCIe are inferred from the expert size and the routing, not counted.

## Run and platform

- **x16 VM:** run `20261016T170556Z`, started `2026-10-16T17:05:56Z`, two rows, both succeeded. One RTX 5090 (32 GB,
  driver 580.159.03, PCIe 5.0 x16) in a VM on a Ryzen 7 7800X3D with 7 cores and 23 GB of RAM, Ubuntu 24.04.1, kernel
  6.8.0. Image id `sha256:9ebbe0109237…`. Archive `results_rtx5090x1_x16.tar.gz`, root `2026-10-16_17-05-56/`.
- **Dev box:** run `20261016T060644Z`, started `2026-10-16T06:06:44Z`, two rows, both succeeded. One RTX 5090 (32 GB,
  driver 580.178.04, PCIe 5.0 x1) on Ubuntu 24.04 with kernel 7.0.0, shared with a desktop session that held about 5
  GB of the card. Image id `sha256:342ffc5f59fa…`. Archive `results_rtx5090x1.tar.gz`, root `2026-10-16_06-06-44/`;
  its benchmark logs also show the dropped third row's failure.

Both archives are in Git LFS and hold the experiment records, the benchmark logs, and per-row `bench_c1.json`,
`image.json`, `responses.jsonl` and `server.log`. Node addresses in the raw files read `node`.
