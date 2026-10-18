# OLMoE-1B-7B-0125-Instruct with uncached experts run on the CPU (strategy G), RTX 5090 at PCIe 5.0 x16

## Result

Running uncached experts on the CPU does not beat milestone 3's cache at one stream. It helps at eight streams, but
less than the table's headline suggests. At concurrency 1 the cache wins at every budget: taking decode off the CUDA
graph and reading each layer's routing to the host costs 2–6 ms per token on its own, and every CPU expert then adds
time instead of saving it. At concurrency 8 strategy G is 1.8–5.2x faster than the cache, but most of that is the
cache's own gap: at width 8 it reads every expert over the link. Against the same eager path with no CPU experts,
sending half of a layer's uncached experts to the CPU is 0.87–1.27x, a win in four of six cells.

TPOT median over 8 requests, ms; TTFT median in brackets:

| GPU experts | Row | Chat c1 | Code c1 | Chat c8 | Code c8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 48 | cache, decode graph | **5.99** (302) | **5.15** (274) | 173.09 (1069) | 146.15 (842) |
| | G, no CPU experts | 11.29 (123) | 11.05 (93) | **33.17** (329) | 35.27 (386) |
| | G, half on the CPU | 15.00 (118) | 13.88 (88) | 33.65 (469) | **28.47** (265) |
| | G, all on the CPU | 17.78 (118) | 19.27 (113) | 43.70 (425) | 32.75 (259) |
| 32 | cache, decode graph | **10.46** (311) | **8.19** (274) | 171.31 (1095) | 145.98 (904) |
| | G, no CPU experts | 14.25 (198) | 12.70 (162) | 71.23 (653) | **53.35** (461) |
| | G, half on the CPU | 19.00 (192) | 17.27 (157) | **59.13** (616) | 61.18 (565) |
| | G, all on the CPU | 28.11 (199) | 29.85 (181) | 75.67 (602) | 59.06 (495) |
| 16 | cache, decode graph | **18.73** (308) | **14.60** (278) | 171.35 (1059) | 147.76 (827) |
| | G, no CPU experts | 20.90 (272) | 17.05 (243) | 121.56 (809) | 95.09 (774) |
| | G, half on the CPU | 31.61 (274) | 28.95 (233) | **95.39** (805) | **76.20** (721) |
| | G, all on the CPU | 40.48 (284) | 41.30 (247) | 120.54 (793) | 102.20 (536) |

"Half" and "all" are the share of a layer step's uncached experts with at most 2 rows that may run on the CPU
(`EMMY_GEN_EXPERTS_CPU=2`, `EMMY_GEN_EXPERTS_CPU_SHARE=0.5` or `1`). A repeat of the 16-expert half row in a separate
server boot gave 25.38 / 30.31 / 93.16 / 77.36 ms: concurrency 1 varies by up to 6 ms between boots, concurrency 8
within 2 ms.

**Why the CPU does not pay at one stream.** Two measured facts bound it on this machine:

- The CPU is no faster than the link. A one-off probe (not kept) on the same VM read one cold expert in 260 us on
  the CPU backend (48 GB/s) and copied it over the link in 242 us (52 GB/s); both at once, on different experts,
  read 69 GB/s together, so the best a split can do is 1.34x on the bytes of the uncached experts. In the server a CPU
  expert costs about 375 us, measured from TPOT; why it is slower than alone is not resolved. Memory compaction is
  not the cause: turning it off changed nothing.
- An expert run on the CPU is not cached. The cache wins at one stream by copying every miss in, so the next steps
  hit it. A CPU expert is never copied, so it misses again, and at one stream that costs more than the overlap saves.
  Copying it in anyway would put its bytes back on the link, which is what bounds the step.

**The host round trip (M2).** "G, no CPU experts" runs every expert on the GPU through the same cache, but eager and
with one host read of the routing per layer. At concurrency 1 it costs 2.2–2.5 ms per token at 16 GPU experts and
5.3–5.9 ms at 48, against the captured cache: 0.14–0.37 ms per layer for the lost graph and the read together.

**Prefill gains, but not from the CPU.** TTFT is 1.6–2.9x lower on every strategy G row at 48 and 32 experts because the
routed path reads cached experts from their device slots. Milestone 3's cache reads every expert from host memory in
prefill; this is a fix to that path, not to CPU experts.

**Tokens.** Every cache row gave the same 8 chat and 8 code answers at concurrency 1. No strategy G row is token for
token identical to them, not even the one with no CPU experts: the routed combine adds the experts' outputs in a
different order and precision from the fixed-slot one, and the CPU kernels round differently again. 4–7 of 8 answers
match exactly, and the ones that do not are coherent alternatives that part from the reference at a near-tie,
sometimes in the first words. Concurrency 8 is not deterministic across rows on any path, the cache's included.

What this means for the plan:

- **Strategy G earns its place only at batch widths, and on a machine whose CPU reads memory faster than its link.**
  This VM's 7 cores read 48 GB/s; a server reading 300 GB/s would change the one-stream answer. Measure there before
  designing G in.
- **The bigger win in this table is the cache's own wide-step path.** At concurrency 8 the eager cache with no CPU
  experts is 1.4–5.2x faster than milestone 3's, because it uses its device slots for every width. The design
  (milestone 6) should give the cache that path whatever happens to G.
- **A one-stream step should keep its CUDA graph.** The 2–6 ms the host read costs is more than any CPU expert saves
  here; G at one stream needs its decision on the device, or none.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/cpu_experts_rtx5090 --ssh …` served
`allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's
vLLM 0.23 plugin, from the image built at commit `b3e851535`. The strategy G code (`EMMY_GEN_EXPERTS_CPU`,
`EMMY_GEN_EXPERTS_CPU_SHARE`) was not merged: it exists at that commit and in that image only. Serving shape, strict evidence, `rank.json` and the
prompts are those of hot_set_rtx5090; each row served 8 chat and 8 code prompts at concurrency 1 and again at 8, 64
output tokens, `--ignore-eos`, temperature 0, with `--max-num-seqs 8`. The cache rows capture single-token decode
(`FULL_DECODE_ONLY`, size 1), as in milestone 3; the strategy G rows run eager. Each row is one server boot.

Strategy G per MoE layer step: the router's picks are read to the host; an expert the cache holds runs on the GPU
from its slot; of the uncached experts with at most 2 rows, the share set by the row, fewest rows first, run on the
CPU from the cache's pinned host copy, through the expert program compiled by the CPU backend (one thread per
physical core but one, the catalog's first tile). The GPU takes the rest at the same time: each is copied into the
slot of the least recently routed expert this step does not read, or read in place over the link when no such slot
is left. Only the CPU experts' rows and outputs cross the link.

## Limitations

- One model, one machine: 7 cores of a Ryzen 7 7800X3D whose memory the CPU reads at about the link's rate.
- The all-GPU row was not run on this image; the tokens are compared with the cache rows, which milestone 3 showed
  give the all-GPU answers.
- The CPU share and the 2-row limit are a fixed rule, two points of a sweep, not a cost model.
- Why a CPU expert runs slower inside the server than alone is not resolved.

## Run and platform

Runs `2026-10-17_04-10-09` (the repeat of the 16-expert half row), `2026-10-17_04-22-14` (16 experts) and
`2026-10-17_04-55-34` (48 and 32), thirteen rows, all succeeded. One RTX 5090 (32 GB, PCIe 5.0 x16) in a CloudRift
VM on a Ryzen 7 7800X3D with 7 cores and 23 GB of RAM plus a 32 GB swap file, the VM of milestone 3.
`results_rtx5090x1_x16.tar.gz` (Git LFS) holds the three run roots: experiment records, benchmark logs, and per row
the four bench JSONs, `image.json` and `server.log`. Node addresses read `node`.
