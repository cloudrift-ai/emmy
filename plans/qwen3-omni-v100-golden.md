# Qwen3-Omni-30B-A3B on V100: from a traced inventory to a golden

Status: open, 2026-10-07. PR #1096 landed what tracing the whole model needs. Every path traces and lowers for
sm_70, but no golden is committed: on a V100 SXM3 32GB almost every program with a matrix multiply runs 100× to
10,000× slower than `torch.compile`, and the compiler cannot yet record the routes that fix it. This plan says what
blocks the golden, in the order to fix it, and how to resume the onboarding afterwards.

Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695`, FP16 on sm_70, standard lane
(`FAST_MATH: false`, as the other V100 goldens).

## The inventory

37 programs, one per distinct path. The thinker's 48 layers, the talker's 20, the audio encoder's 32 and the vision
encoder's 27 each share one layout, so one layer stands for each stack.

| Part | Programs | Shape |
|---|---|---|
| Thinker | MoE decoder layer (`--decoder thinker.model`), embedding, norm + LM head | 512 tokens |
| Audio encoder | three stem convolutions + GELU, stem projection, windowed layer (five 104-token windows), output MLP | 520 tokens |
| Vision encoder | patch embedding, block, merger, deepstack merger | 1024 patches |
| Talker | MoE layer with gated shared expert (`--decoder talker.model`), projection MLP, embedding, head | 512 tokens |
| Code predictor | dense layer (`--decoder talker.code_predictor.model`), embedding, head | 16 tokens |
| code2wav | transformer layer (`--decoder code2wav.pre_transformer`, window 72), code embedding, norm, two upsamplers + ConvNeXt blocks, decoder input conv, four decoder blocks (transposed conv + three residual units), output conv | one 325-frame chunk |

Not traced: the audio encoder's positional add and valid-row gather, and the vision encoder's position-embedding
interpolation (host-side index work).

The decoder layers come from `emmy trace`:

```bash
R=Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695
for spec in thinker.model:512 talker.model:512 talker.code_predictor.model:16 code2wav.pre_transformer:325; do
  emmy trace "$R" --decoder "${spec%:*}" --layer 0 --seq-len "${spec#*:}" --target sm_70 -o work.json --append
done
```

Everything else is an `emmy trace -c` of the Transformers module built from the pinned config, appended with
`--model-provenance "$R" -o work.json --append`. The preamble every `-c` shares:

```python
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as Q

c = AutoConfig.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct", revision="26291f793822fb6be9555850f06dfe95f2d7e695")
for sub in (c.thinker_config.audio_config, c.thinker_config.vision_config, c.code2wav_config):
    sub._attn_implementation = "sdpa"
torch.manual_seed(0)
h = torch.float16
import numpy as np


class Windowed(nn.Module):  # packed windows: cu_seqlens stays a host constant, as the encoder computes it
    def __init__(self, m, cu_seqlens):
        super().__init__()
        self.m = m
        self.cu = np.array(cu_seqlens)

    def forward(self, x, *rest):
        out = self.m(x, self.cu, *rest)
        return out[0] if isinstance(out, tuple) else out
```

and the final expressions, one per program (`$L` / `$C` follow the decoder's lengths and widths: L = 1300, then
`(L - 1) * r` per block with r = 8, 5, 4, 3; C = 1536 halving per block):

```python
Windowed(Q.Qwen3OmniMoeAudioEncoderLayer(c.thinker_config.audio_config), [0, 104, 208, 312, 416, 520]).to(h).eval()(torch.randn(520, 1280, dtype=h))
nn.Sequential(e.ln_post, e.proj1, e.act, e.proj2).to(h).eval()(torch.randn(520, 1280, dtype=h))  # e = Q.Qwen3OmniMoeAudioEncoder._from_config(audio_config)
Conv(e.conv2d{1,2,3}) + F.gelu over (40, 1, 128, 100), (40, 480, 64, 50), (40, 480, 32, 25)  # one program each
e.conv_out over x.permute(0, 3, 1, 2).view(b, t, c * f) from (40, 480, 16, 13)
Windowed(Q.Qwen3OmniMoeVisionBlock(vision_config), [0, 1024])(torch.randn(1024, 1152), (cos, sin) each (1024, 72))
Q.Qwen3OmniMoeVisionPatchMerger(vision_config, use_postshuffle_norm={False,True})(torch.randn(1024, 1152))
Q.Qwen3OmniMoeVisionPatchEmbed(vision_config)(torch.randn(1024, 1536))
nn.Sequential(Q.Qwen3OmniMoeTextRMSNorm(2048, eps=1e-6), nn.Linear(2048, 152064, bias=False))(torch.randn(1, 512, 2048))
nn.Embedding(152064, 2048)(torch.randint(0, 152064, (1, 512)))
Q.Qwen3OmniMoeTalkerResizeMLP(c.talker_config)(torch.randn(1, 512, 2048))
nn.Sequential(Q.Qwen3OmniMoeTextRMSNorm(1024, eps=1e-6), nn.Linear(1024, 3072, bias=False))(torch.randn(1, 512, 1024))
nn.Embedding(3072, 1024)(torch.randint(0, 3072, (1, 512)))
nn.Sequential(Q.Qwen3OmniMoeRMSNorm(1024, eps=1e-6), nn.Linear(1024, 2048, bias=False))(torch.randn(1, 16, 1024))
nn.Embedding(2048, 1024)(torch.randint(0, 2048, (1, 16)))
code_embedding(codes + code_offset).mean(1) over torch.randint(0, 2048, (1, 16, 325))
Q.Qwen3OmniMoeRMSNorm(1024, eps=1e-5)(torch.randn(1, 325, 1024))
Q.Qwen3OmniMoeCausalTransConvNet(1024, 1024, 2, 2) over L = 325, 650; Q.Qwen3OmniMoeConvNeXtBlock(1024) over 650, 1300
Q.Qwen3OmniMoeCausalConvNet(1024, 1536, 7)(torch.randn(1, 1024, 1300))
nn.Sequential(Q.Qwen3OmniMoeSnakeBeta(C), Q.Qwen3OmniMoeCausalTransConvNet(C, C // 2, 2 * r, r)) over (1, C, L)
nn.Sequential(*[Q.Qwen3OmniMoeCode2WavDecoderResidualUnit(C // 2, d) for d in (1, 3, 9)]) over (1, C // 2, (L - 1) * r)
SnakeBeta(96) -> CausalConvNet(96, 1, 7) -> clamp(-1, 1) over (1, 96, 623445)
```

Every module is `.to(h).eval()` and every input `dtype=h`. The working golden's header must then name the card
(`gpu_name: NVIDIA Tesla V100 SXM3 32GB`) and every row pin `FAST_MATH: false`: the trace stamps the dev box's GPU
and the default lane, and `--record` refuses a file seeded for another card.

## Where it stands on the V100

First pass, greedy pick from an empty tune DB (fast-math lane, before the header fix), Emmy against `torch.compile`
(µs):

| Program | Emmy | torch.compile |
|---|---|---|
| embeddings (thinker, talker, code predictor) | 1.5 to 3.5 | 2.4 to 4.6 |
| audio output MLP | 1,092,755 | 98 |
| vision merger / deepstack merger | 125,527 / 3,002 | 319 / 314 |
| talker norm + codec head | 55,685 | 68 |
| code predictor layer | 561,524 | 149 |
| code2wav decoder input conv / output conv | 364,097 / 6,124 | 515 / 1,571 |
| code2wav 2× upsampler / block-1 transposed conv | 18,239 / 593,144 | 141 / 1,402 |
| thinker, talker, code2wav layers; thinker head; audio stem; two residual blocks | hung, out of memory, or over the 60 s run budget | |

A plain 520×1280×1280 linear is 66 µs against 36 µs, on the tensor-core tier. The losses are the kernel sets the
greedy takes, not the schedules it can reach.

On the code predictor layer, a hand-chosen route (all eight contraction seams cut as one composed decision, every
weight layout pinned `source`) took 562 ms to 7.0 ms, and `--kernel … --tune 24` took its four slowest pieces from
2,014 / 1,029 / 858 / 160 µs to 57 / 43 / 225 / 137 µs alone (`torch.compile`: 150 µs for the layer).

## What blocks the golden, in order

1. **A measured route outranks every unmeasured one.** At a kernel-set fork the greedy takes the fastest measured arm
   (`policy/greedy.py`, `kernel_set`) and consults the placement prior only when no arm is measured. The first bench
   of a new shape measures the prior's pick, so a 1 s fused kernel is then preferred forever over cuts nobody
   measured. Fix: price unmeasured arms too (the placement prior, or the sum of the pieces' schedule-prior
   predictions) and compare, or have `run --bench` measure the fork's arms rather than only the one it took. This is a
   pricing change: run the whole passes lane before and after.
2. **Nested decisions cannot be priced or recorded.** A composed cut whose seams sit in one another's cones realizes a
   piece of a piece (`…__place_57c259763f__place_d9d2185645`), and a weight-layout choice leaves a kernel that forks
   on the next weight's layout. The intermediate kernel gets no row, so its fork has no measured arm:
   `--record-greedy` (strict evidence) refuses with `no measured row spells a kernel-set arm`, and an unpinned
   compile never takes the route again. Only pinning every layout (`LAYOUT=source`) let the route record, and it
   still replays only under its pins. Fix in the engine, not per pass: price a nested arm through its terminal
   pieces, or record the chain as one parent route (see the PR #1064 flattening).
3. **A tuned piece is not what its layer runs.** Piece `k_sdpa_linear_mean_reduce_d5e473__place_8cdae16257` tuned to
   225 µs as `--kernel`, but the layer under the same pins ran it at 6.7 ms. Find whether the layer's schedule fork
   keys the piece differently from the stored body `--kernel` builds (bindings, io names, layout), and add a test
   that a tuned `--kernel` row is the one the layer compile picks.
4. **Stacked convolutions cut into an impossible workspace.** Traced as one program, the audio stem's three 3×3
   stride-2 convolutions fuse into one region; the greedy's cut materializes conv2's im2col window in conv3's im2col
   coordinates, a 1.6e15-element workspace. The cut pass should not offer a seam whose workspace exceeds device
   memory by orders of magnitude. The stem is traced one program per convolution until then.
5. **An out-of-memory provisioning failure blames no kernel.** It records no `bench_fail` row, so the next compile
   repeats it. Blame the kernel that owns the oversized workspace.
6. **Kernel-set pricing dominates compile time.** On the rented V100's 4 cores, a layer program spends 15 to 75
   minutes, almost all in `canonicalize_identity` under `_kernel_set_pick`. Rent a V100 32GB with more cores for the
   recording run, or memoize piece identities across a compile.

## Resuming the onboarding

After 1 to 3, run the `onboard-model` skill on 1× NVIDIA Tesla V100 SXM3 32GB with the inventory above: bench every
program, tune the losers' pieces (`emmy run --golden work.json --kernel <piece> --bench --tune 100`), re-bench each
whole program, record, and hold every target to the `torch.compile` bar. Seam spellings for a pinned route can be read
by wrapping `passes.tile._cut.cuttable_seams` around one `emmy compile --golden work.json --program N --target sm_70`
(a pinned compile takes seconds, not minutes). Commit `recipes/Qwen3-Omni-30B-A3B-Instruct/golden/v100_sm70.json`
only when every program is covered and at par. Serving the model (vLLM's thinker on the 1Cat fork, speech output
outside vLLM) is a separate decision after the golden.
