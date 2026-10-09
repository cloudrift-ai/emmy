# NVFP4 tuning follow-ups

## Realize a nested contraction's selected tile under a serial carrier

A selected native tile on a nested contraction is not realized when an enclosing serial carrier lowers the
whole subtree with `Fold.lower`. The CUDA can compute correct values while retaining TILE/STAGE/WORK stamps
for a schedule it never executes. Scalar register tiles can be discarded in the same way.

Known instances:

- Qwen3.8 NVFP4 program 0 gate/up pieces `k_linear_reduce_139b4a` and
  `k_linear_reduce_139b4a__place_6d4a2fe9d9__place_eff42e202f`, under native FP4
  TILE=f1x2/k4, STAGE=d2/smem-async, WORK=w4x1.
- `attention/sdpa-hd128-softmax-v-mma.json`: the serial softmax/value carrier discarded the score's native
  FP16 tile and async stage. Its corrected corpus row describes the scalar CUDA that actually ran.
- `attention/rmsnorm-qk-sdpa-stat-cut.json`: the serial attention carrier discarded the score's scalar f1
  tile and its t256 work assignment. Its corrected row also preserves the existing scalar CUDA.
- `attention/rmsnorm-gqa-sdpa-stat-fill.json` on sm80: the carrier discarded the score's native FP16 tile
  and synchronous stage. Its corrected scalar row preserves the existing CUDA.
- The paged/flat attention comparison under STAGE=d2/smem: the carrier discarded the nested f4x4 tile and
  stage. The comparison now explicitly selects scalar direct loads.
- The sm120 serving-test fixture rows `g003.k_linear_mean_reduce_7defda.f14a71b7cee7` and
  `g037.k_conv1d_linear_mean_reduce_c7f4f6.6f3ef43a1ade`: their corrected TILE/STAGE values are OFF. The
  latter retains its t32 worker inventory for its cooperative reduction; the former uses scalar workers.
  Both corrections preserve the original CUDA source byte for byte.

The scalar binder refuses a selected contraction tile anywhere in the subtree it would lower serially.
The existing rejected-row path tries another schedule; a pin with no realizable alternative fails lowering,
and strict evidence rejects an unmeasured substitute. This prevents those stamps from qualifying scalar code
as a realized native or staged row.

A real implementation must bind the nested contraction inside the carrier's iteration, preserve the carrier's
reduction order and ownership of intermediate values, and consume the selected tile, worker inventory, and
transport. Tests must assert the emitted native/staged instructions as well as numerical correctness. The
current scalar corpus oracles remain useful; native coverage needs separate authored rows once it exists.

Other serving fixtures with the same ignored nested tiles now describe the original CUDA byte for byte:

- sm70: `g014.k_linear_mean_reduce_b18cbc`.
- sm80: `g003.k_linear_mean_reduce_7defda.f14a71b7cee7`,
  `g035.k_linear_matmul_mean_reduce_e38648.129979d8a9ee`,
  `g036.k_linear_matmul_mean_reduce_04e3b3.ccc052d17996`,
  `g038.k_linear_matmul_mean_reduce_a995d4.35d41124e1e9`.
- sm90: `g035.k_linear_matmul_mean_reduce_e38648.129979d8a9ee`,
  `g036.k_linear_matmul_mean_reduce_04e3b3.ccc052d17996`,
  `g038.k_linear_matmul_mean_reduce_a995d4.35d41124e1e9`,
  `g039.k_linear_matmul_mean_reduce_189167.15812746a3b3`.

Their TILE/STAGE values are OFF. Cooperative reductions retain their original worker inventories.

## Complete authored serving rows before schedule replay

Five serving fixture rows lack required classic schedule keys and cannot be decoded. Their schedule behavior
cannot be established by the selected-row audit; they remain unchanged and need complete authored rows:

- sm70: `g008.k_linear_reduce_00d191.2a5b4135f132` (REDUCE/STAGE/TILE keys),
  `g035.k_linear_matmul_mean_reduce_e38648.048dcd8c1ee8` (STAGE/TILE keys).
- sm80 and sm90: `g008.k_linear_reduce_00d191.ec5e55ae07bc` (REDUCE/STAGE/TILE keys on each card).
- sm120: `g040.k_conv1d_linear_mean_reduce_c59d7d.cddc7092cef7` (STAGE/TILE keys).
