# Hand pins cannot select a cut piece’s WORK; attention compilation is slow

## Summary

A pin fixes a compiler choice. After a cut splits a fused kernel into pieces, global pins can make their schedules
conflict: a thread-layout pin helps an output writer but removes tensor-core schedules from its matmuls. A CPU-only
follow-up confirms that existing site-scoped pins do not select the writer’s `WORK`: that choice is kernel-level and
read only under the bare key. Separately, the attention kernel takes minutes to compile without pins, although the
recorded concurrent runs do not isolate where that time goes.

## Observed and desired Tile IR

Observed with `OPROJ_CUT` alone at `a98fd4f8`: the projection has a tensor-core TILE, but the residual writer has no
`work` line and leaves `a3` as a serial output sweep. These are excerpts from two pieces of the same saved dump:

```text
=== 5: k_linear_mean_reduce_fd2717__place_6abfbbaa03 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   w1x4
    Fold[a2 in 0..6144] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f4x2/k8 STAGE=d1/smem-async/p2⟩
    …

=== 6: k_linear_mean_reduce_fd2717__place_a7de42f3e4 ===
    place  free=(a0)  grid=(a0)
    Fold  free
    ├─ operand[acc2__ws6abfbbaa03]: load add_8__place_6abfbbaa03_0[a0, a3]   ‹materialized›
    └─ lift: λ(acc2__ws6abfbbaa03, a0, a3) -> (v31__ws6abfbbaa03)
         f16 v30__ws6abfbbaa03 = copy(acc2__ws6abfbbaa03)
         in22 = load hidden_states[0, a0, a3]
         v31__ws6abfbbaa03 = add(in22, v30__ws6abfbbaa03)
    outputs
    └─ sweep(a3) add_8[0, a0, a3] = v31__ws6abfbbaa03
```

Desired combination, **composed, not emitted**: retain that projection schedule and give the writer a thread layout.
Its arithmetic is unchanged and omitted here:

```text
=== 5: k_linear_mean_reduce_fd2717__place_6abfbbaa03 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   w1x4
    Fold[a2 in 0..6144] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f4x2/k8 STAGE=d1/smem-async/p2⟩
    …

=== 6: k_linear_mean_reduce_fd2717__place_a7de42f3e4 ===
    place  free=(a0, a3)  grid=(a0, a3)
    work   t128
    Fold  free
    …
    outputs
    └─ add_8[0, a0, a3] = v31__ws6abfbbaa03
```

The global `WORK=t128` attempt instead removes tensor-core schedules from the projection pieces, as described below.
The follow-up below establishes why existing site-scoped `WORK` pins do not produce this combination.

## Observed details

1. **A global schedule pin affects pieces it was not intended for.** After the cut pass splits a fused kernel into
   pieces, each piece forks over its own schedule: TILE, STAGE, WORK and REDUCE. A pin without a site (`TILE=…`,
   `WORK=…`) is published to every piece. Node-scoped TILE/REDUCE/STAGE keys address sites inside each piece’s
   Fold tree; WORK has no node scope, and none of these keys includes a piece selector. Global publication is intended
   behavior; the missing capability is restricting a hand pin to the desired piece.
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
   recipe that combines the desired schedules without affecting unrelated pieces. The tested WORK scopes do not
   provide it; this does not invalidate the existing per-piece golden mechanism.
2. **The unpinned tile compile of the attention kernel takes about 12 minutes.**
   - `emmy compile --golden … --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile` took 714–717 s at 16 tokens,
     and 742 s at 512. These ran 4 compiles in parallel on the same machine.
   - With the full-projection cut pinned, the same compile takes 71.5 s. With the cut and an fp4 TILE pinned, it takes
     13.7 s.
   - The reduction with pins suggests fork search is a major contributor. It does not isolate the cause: these runs
     shared the machine, and no pass profile was collected. `-v` prints only the total ("compile: total 714.29s
     (deterministic resolve)").

## CPU-only follow-up: the WORK scope is missing

At `a98fd4f8`, the actual layer-3 o_proj graph was cut with `OPROJ_CUT`, using `--passes tp`. Its residual writer
`…__place_a7de42f3e4` has no TILE or REDUCE sites. Its kernel WORK catalog includes the empty layout and `t4` through
`t512`. Inspecting `ClassicProblem.kernel_site` for that piece and the projection `…__place_6abfbbaa03` gives:

| Supplied row | Residual writer's WORK choices | Projection's WORK choices |
| --- | --- | --- |
| No WORK pin | Empty or thread layouts | Empty, thread or warp layouts |
| `WORK=t128` | Only `t128` | Only `t128` |
| `WORK@n0=t128` | Same as no pin | Same as no pin |
| `WORK@<piece-name>=t128` | Same as no pin | Same as no pin |

This follows directly from `ClassicKernelSite._works` in `emmy/compiler/ir/schedule/classic/sites.py`: it reads
`row.get("WORK")`. The schedule codec also emits only bare WORK. The four projection pieces each expose their sole
contraction under the same bare TILE key; node sites are local to a piece, not identifiers for pieces.

A fresh CLI compile with `OPROJ_CUT,WORK@k_linear_mean_reduce_fd2717__place_a7de42f3e4=t128`, `--passes tph` and
`--ir tile` completes but leaves the writer unchanged. Actual emitted Tile IR:

```text
=== 4: k_linear_mean_reduce_fd2717__place_a7de42f3e4 ===
    place  free=(a0)  grid=(a0)
    Fold  free
    …
    outputs
    └─ sweep(a3) add_8[0, a0, a3] = v31__ws6abfbbaa03
```

There is still no `work` line. The neighboring projection retains an fp4 tensor-core TILE, but the requested writer
layout was not selected. This establishes a hand-targeting gap for WORK without assuming a new syntax is the right
fix. The attention TILE-targeting example was not separately exhausted, and the compile-time cause remains unverified.

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

# 1a follow-up: a piece name is not a supported WORK scope; the writer remains serial.
EMMY_KNOBS="$OPROJ_CUT,WORK@k_linear_mean_reduce_fd2717__place_a7de42f3e4=t128" \
  ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_linear_mean_reduce_fd2717 \
  --target sm_120 --passes tph --ir tile

# 1b. Related RMSNorm + encode writer with WORK=t128 (not a timing guarantee for o_proj):
EMMY_KNOBS='PLACE@map.1/map.2/reduce.3/map.1/reduce=cut,PLACE@map.1/map.2/reduce=cut,PLACE@map.2/map.2/reduce=cut,WORK=t128' \
  ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_mean_01c219 --bench --no-record-evidence

# 2. Unpinned tile compile time:
time ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels /tmp/l3_s16.golden.json` prints
each kernel's Loop IR as one JSON line, and its `"name"` fields list the current names.

## Fix criteria

1. **Targeted hand pins.** A documented and tested way to restrict a pin to a piece gives every o_proj
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
