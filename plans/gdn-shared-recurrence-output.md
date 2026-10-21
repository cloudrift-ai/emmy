# GDN shared recurrence output: follow-ups

The cut pass now treats the GDN recurrence output, read flat, by blocks of sixteen and by packed pairs, as one value
with one workspace. That changed the kernel sets of the width-16 and width-64 GDN programs, so their recorded routes
no longer replay. The RTX 5090 NVFP4 golden was re-recorded for them. The items below are what is still open.

## 1. V100 FP8 gdn16 has no route

V100 FP8 gdn16 (the V100 decode program, decode bucket 16, no width-1 tier) has no route after this change; a strict
V100 deploy refuses it until a batched V100 re-measure records it.

The restamp of `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` dropped the 25-cut decision on
`k_linear_matmul_mean_reduce_7ffa7e` and the 3-cut decision nested under it (2 routing rows, 12 -> 10), the 19
kernels those decisions minted (68 -> 49), and their 19 rows (74 -> 55; gdn16 rows 25 -> 6). No kernel was re-keyed.

To restore it on a V100: pin the old route plus the full-projection cut of the new sibling-producer pieces (the two
contractions that share a workspace dependency now fuse into one piece whose sweeps the grid cannot bind), record
with `emmy run --golden … --record-greedy` after one non-strict bench has put evidence for every fork into the tune DB,
then tune the new pieces. On the RTX 5090 that took gdn16 from 408 µs untuned to 98 µs.
