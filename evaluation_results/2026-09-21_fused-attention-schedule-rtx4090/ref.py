"""PyTorch eager reference for the same target, measured in the same place as the CUDA variants.

Same shapes, same maths, same checksum reductions (sum of |score| and sum of score^2) so the
numbers are comparable to qk_scores.cu rather than to a different run's table.
"""

import statistics

import torch

HQ, HKV, S, D = 16, 8, 512, 128
DH = D // 2
dev = "cuda"

torch.manual_seed(0)
Q = torch.empty(HQ, S, D, device=dev, dtype=torch.float16).uniform_(-1, 1)
K = torch.empty(HKV, S, D, device=dev, dtype=torch.float16).uniform_(-1, 1)
gq = torch.ones(D, device=dev, dtype=torch.float32)
gk = torch.ones(D, device=dev, dtype=torch.float32)
pos = torch.arange(S, device=dev, dtype=torch.float32)
inv = 1.0 / (10000.0 ** (2 * torch.arange(DH, device=dev, dtype=torch.float32) / D))
th = pos[:, None] * inv[None, :]
cosv, sinv = torch.cos(th), torch.sin(th)


def rope(x):
    lo, hi = x[..., :DH], x[..., DH:]
    return torch.cat([lo * cosv - hi * sinv, hi * cosv + lo * sinv], dim=-1)


def rmsnorm(x, g, eps=1e-6):
    f = x.float()
    return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * g


def step():
    qn = rope(rmsnorm(Q, gq)).half()
    kn = rope(rmsnorm(K, gk)).half()
    kn = kn.repeat_interleave(HQ // HKV, dim=0)  # GQA 2:1
    sc = torch.matmul(qn.float(), kn.float().transpose(-1, -2))
    return sc.abs().sum(-1), sc.pow(2).sum(-1)


def bench(fn, warm=5, iters=11):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
print(f"shape Q[{HQ},{S},{D}] K[{HKV},{S},{D}] f16, GQA {HQ // HKV}:1\n")

eager_ms = bench(step)
ref_abs, ref_sq = step()

compiled = torch.compile(step, fullgraph=True)
try:
    ind_ms = bench(compiled)
    c_abs, c_sq = compiled()
    torch.cuda.synchronize()
    d = (c_abs - ref_abs).abs().max().item() / (ref_abs.abs().max().item() + 1e-6)
    ind_note = f"agrees with eager to {d:.2e}"
except Exception as exc:  # Inductor fails on this target in the recorded runs; say so rather than hide it
    ind_ms, ind_note = float("nan"), f"FAILED: {type(exc).__name__}: {str(exc)[:80]}"

print(f"{'lane':22s} {'median ms':>10s}  note")
print(f"{'eager PyTorch':22s} {eager_ms:10.3f}  reference")
print(f"{'torch.compile':22s} {ind_ms:10.3f}  {ind_note}")
print(f"\nchecksum row0: sum|score| = {ref_abs[0, 0].item():.4f}  sum score^2 = {ref_sq[0, 0].item():.4f}")
