# Global pins interfere across cut pieces; attention compilation is slow

## Summary

A pin fixes a compiler choice. After a cut splits a fused kernel into pieces, global pins can make their schedules
conflict: a thread-layout pin helps an output writer but removes tensor-core schedules from its matmuls. Existing site-scoped pins may already solve this;
that has not been tested. Separately, the attention kernel takes minutes to compile without pins, although the
recorded concurrent runs do not isolate where that time goes.

## Reading the pin interference

Normalized Tile IR schedule summary; instruction and piece names are abbreviated:

```text
Observed with global WORK=t128:     Desired combination (illustrative):
residual writer: work t128          residual writer: work t128
projection: work t128; TILE=f1      projection: TILE=mma_m16n8k64_e2m1_f32/…
            or no TILE                         with a compatible WORK
```

The writer and matmul need different schedules. Whether existing site-scoped pins can express this combination is
unverified; the example does not establish a need for new pin syntax.

## Observed details

1. **A global schedule pin affects pieces it was not intended for.** After the cut pass splits a fused kernel into
   pieces, each piece forks over its own schedule: TILE, STAGE, WORK and REDUCE. A pin without a site (`TILE=…`,
   `WORK=…`) applies to every piece. A site-scoped pin (`KNOB@<site>=…`) names a node inside a Fold tree; there is no
   explicit piece selector. Whether existing site-scoped pins can distinguish the pieces below has not been tested.
   Two examples from `Inferact/Qwen3.8-27B-NVFP4` layer 3 at 16 tokens, both after the full-projection cut:
   - **o_proj + residual + RMSNorm + encode** (`k_linear_mean_reduce_fd2717`). The residual-add piece that writes the
     kernel's output (`…__place_a7de42f3e4`) is offered thread layouts `WORK=t4` to `t512`. With the empty layout it
     runs one thread per row and loops over the 5120 hidden elements serially: 361.5 µs, 39% of the kernel.
     `WORK=t128` is the pin that would spread it. But a program-wide `WORK=t128` also lands on the o_proj matmul
     pieces, which then leave the tensor cores (`work t128`, `TILE=f1` or no TILE). A related kernel illustrates the
     potential benefit: in the RMSNorm + encode kernel `k_mean_01c219`, with its statistic and amax cut, `WORK=t128`
     takes its output-writing piece from 362 µs to 5.1 µs (no correctness check yet).
   - **Attention** (`k_sdpa_linear_mean_reduce_c6c239`). Pinning `TILE=mma_m16n8k64_e2m1_f32/f1x1/k8` puts the
     projection pieces on the fp4 cell. The QK score piece cannot take that tile and falls back to the scalar tier:
     242 µs, where the default pick puts it on `mma_m16n8k16_f16_f32/f1x8/k2` at 37 µs. At 512 tokens, a TILE pin sent
     some pieces to 119–318 ms.

   Goldens can hold per-piece rows: a child-identity schedule receipt (`GLOSSARY.md`) decorates exactly one kernel of
   a set, and `emmy run --record-greedy` or `--pin-route` records it. The missing result is a demonstrated hand-pin
   recipe that combines the desired schedules without affecting unrelated pieces.
2. **The unpinned tile compile of the attention kernel takes about 12 minutes.**
   - `emmy compile --golden … --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile` took 714–717 s at 16 tokens,
     and 742 s at 512. These ran 4 compiles in parallel on the same machine.
   - With the full-projection cut pinned, the same compile takes 71.5 s. With the cut and an fp4 TILE pinned, it takes
     13.7 s.
   - The reduction with pins suggests fork search is a major contributor. It does not isolate the cause: these runs
     shared the machine, and no pass profile was collected. `-v` prints only the total ("compile: total 714.29s
     (deterministic resolve)").

## Review remark: test site-scoped pins before adding syntax

The examples establish interference from global pins, not the impossibility of targeting one piece. First inspect the
schedule sites of the cut pieces and try the existing `KNOB@<site>=value` form. If it works, document and test that
recipe; new syntax is unnecessary. If sites collide or are rebound so that the requested schedules cannot coexist,
retain a minimal reproducer of that failure before proposing piece-targeting syntax. This targeting question is
separate from the unexplained compile time.

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB. `emmy trace` writes a layer
inventory; `--realization <name>` selects one kernel. Pins go in `EMMY_KNOBS` as comma-separated
`KNOB=value` or `KNOB@<site>=value` choices.

```sh
M=Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462   # ~26 GB download on first use
rm -f /tmp/q38.db; export EMMY_TUNE_DB=/tmp/q38.db
./venv/bin/emmy trace $M --layer 3 --seq-len 16 --target sm_120 -o /tmp/l3_s16.golden.json

# 1a. o_proj kernel: the cut alone, then the cut with WORK=t128.
OPROJ_CUT='PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,PLACE@map.2/map.2/reduce.4/map.1/reduce=cut,PLACE@map.3/map=cut,PLACE@map.3/map.2/inner=cut,PLACE@map.3/map.3/reduce.1/inner=cut'
EMMY_KNOBS="$OPROJ_CUT" ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_linear_mean_reduce_fd2717 --bench --no-record-evidence
#   bench table: …__place_a7de42f3e4  361.5 µs, grid 1, empty WORK
EMMY_KNOBS="$OPROJ_CUT,WORK=t128" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_linear_mean_reduce_fd2717 --ir tile
#   every piece prints `work t128`; the o_proj pieces carry TILE=f1 or none instead of mma_m16n8k64_e2m1_f32/…

# 1b. Related RMSNorm + encode writer with WORK=t128 (not a timing guarantee for o_proj):
EMMY_KNOBS='PLACE@map.1/map.2/reduce.3/map.1/reduce=cut,PLACE@map.1/map.2/reduce=cut,PLACE@map.2/map.2/reduce=cut,WORK=t128' \
  ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_mean_01c219 --bench --no-record-evidence

# 2. Unpinned tile compile time:
time ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels /tmp/l3_s16.golden.json` prints
each kernel's Loop IR as one JSON line, and its `"name"` fields list the current names.

## Fix criteria

1. **Targeted hand pins.** A documented and tested recipe, using existing site pins if sufficient, gives every o_proj
   piece an fp4-cell TILE and gives the residual-add piece `WORK=t128` in one compile of `k_linear_mean_reduce_fd2717`
   under `OPROJ_CUT`:
   - Tile IR shows `work t128` on the residual-add piece and an `mma_m16n8k64_e2m1_f32/…` TILE on the o_proj pieces.
   - CUDA distributes the residual-add output sweep across the requested threads. Report the writer's time and the
     complete kernel set's time before and after, with a correctness check that actually ran. The related kernel's 5.1
     µs result is context, not a deadline for this different kernel.
   - A pin whose target piece does not exist fails loudly and names the target.
2. **Compile time.** Profile the unpinned tile compile of `k_sdpa_linear_mean_reduce_c6c239` at 16 tokens in
   isolation, identify the expensive pass or search step, and remove the identified avoidable work. Report repeated
   before/after times under the same cache, evidence and hardware conditions, plus the chosen kernel set. Compare with
   the pinned route without removing legal choices to make the search faster. The existing concurrent timings do not
   justify a universal one-minute threshold.
