"""The compiled vision tower on CUDA: a tiny random Qwen3.5 tower's programs against its transformers modules.

``CompiledTower`` takes vLLM's tower for the batch metadata and each block's attention. vLLM's tower needs an
engine, so a stand-in with the same surface takes its place: ``prepare_encoder_metadata`` over a grid, and an
attention whose ``qkv`` and ``proj`` return ``(tensor, bias)`` as vLLM's linears do. The reference runs the same
attention with the transformers projections, the norms, the MLPs and the merger, in bf16.
"""

import pytest

from tests.compiler.helpers import requires_cuda

pytestmark = [pytest.mark.xdist_group("cuda"), requires_cuda]

torch = pytest.importorskip("torch")
nn = torch.nn


class _Tupled(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x), None


class _Attention(nn.Module):
    """vLLM's vision attention surface: the projections, rotary over the full head and attention per image."""

    def __init__(self, attn):
        super().__init__()
        self.qkv, self.proj, self.heads = _Tupled(attn.qkv), _Tupled(attn.proj), attn.num_heads

    def forward(self, x, *, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen, sequence_lengths):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

        x, _ = self.qkv(x)
        rows = x.shape[0]
        q, k, v = x.reshape(rows, 3, self.heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, rotary_pos_emb_cos, rotary_pos_emb_sin)
        out = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
            heads = [t[start:end].transpose(0, 1) for t in (q, k, v)]
            out.append(torch.nn.functional.scaled_dot_product_attention(*heads).transpose(0, 1).reshape(end - start, -1))
        context, _ = self.proj(torch.cat(out).unsqueeze(1))
        return context


class _VllmTower(nn.Module):
    """The surface ``CompiledTower`` reads of vLLM's ``Qwen3_VisionTransformer``, over transformers' modules."""

    def __init__(self, tower):
        super().__init__()
        self.patch_embed, self.merger = tower.patch_embed, tower.merger
        self.blocks = nn.ModuleList(tower.blocks)
        for block in self.blocks:
            block.attn = _Attention(block.attn)
        self.spatial_merge_size, self.patch_size = tower.spatial_merge_size, tower.patch_size
        self.head_dim = tower.config.hidden_size // tower.config.num_heads

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def prepare_encoder_metadata(self, grid):
        """Each image's position table and rotary angles depend on its own grid alone, as vLLM's do."""
        positions, angles = [], []
        for t, h, w in grid:
            generator = torch.Generator().manual_seed(1000 * h + w)
            positions.append(torch.randn(t * h * w, self.patch_embed.embed_dim, generator=generator))
            angles.append(torch.rand(t * h * w, self.head_dim, generator=generator) * 6.0)
        angles = torch.cat(angles)
        sizes = torch.tensor([h * w for t, h, w in grid for _ in range(t)])
        return {
            "pos_embeds": torch.cat(positions).to(self.device, self.dtype),
            "rotary_pos_emb_cos": angles.cos().to(self.device),
            "rotary_pos_emb_sin": angles.sin().to(self.device),
            "cu_seqlens": torch.cat([torch.zeros(1, dtype=torch.int32), sizes.cumsum(0, dtype=torch.int32)]).to(self.device),
            "max_seqlen": torch.tensor(int(sizes.max()), dtype=torch.int32),
            "sequence_lengths": None,
        }


def _reference(tower, x, grid):
    meta = tower.prepare_encoder_metadata(grid)
    hidden = tower.patch_embed(x) + meta["pos_embeds"]
    attention = {key: meta[key] for key in ("cu_seqlens", "rotary_pos_emb_cos", "rotary_pos_emb_sin", "max_seqlen", "sequence_lengths")}
    for block in tower.blocks:
        hidden = hidden + block.attn(block.norm1(hidden).unsqueeze(1), **attention).squeeze(1)
        hidden = hidden + block.mlp(block.norm2(hidden))
    return tower.merger(hidden)


def test_compiled_tower_matches_the_transformers_tower(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers import AutoModel

    from emmy.serving.vision_tower import IMAGE_TOKEN_BUCKETS, CompiledTower
    from tests.serving.generation.test_vision_tower import vl_config

    monkeypatch.setenv("EMMY_GOLDEN_FILE", "")
    config = vl_config()
    torch.manual_seed(0)
    model = AutoModel.from_config(config, dtype=torch.bfloat16)
    config.save_pretrained(tmp_path)
    save_file(
        {f"model.visual.{name}": t.contiguous() for name, t in model.visual.state_dict().items()}, str(tmp_path / "model.safetensors")
    )
    with torch.no_grad():
        tower = _VllmTower(model.visual).cuda()
        # Two small images run together at the narrow width; with a wide one they run at the wide width, and an image
        # as wide as the widest program runs alone.
        small, wide = [[1, 4, 8], [1, 8, 8]], [[1, 32, 64], [1, 64, 64]]
        calls = []
        for grid in (small, small + wide):
            x = torch.randn(sum(t * h * w for t, h, w in grid), 3 * 2 * 8 * 8, device="cuda", dtype=torch.bfloat16)
            calls.append((x, grid, _reference(tower, x, grid)))
        compiled = CompiledTower(tower, str(tmp_path), torch.bfloat16)
        assert set(compiled.programs["pre"]) == set(IMAGE_TOKEN_BUCKETS)
        assert all(len(compiled.programs[piece][tokens]) == 2 for piece in ("pre", "post") for tokens in IMAGE_TOKEN_BUCKETS)
        # vLLM's tower keeps the attention and the metadata; the programs own every block's weights.
        assert not any(hasattr(block, name) for block in tower.blocks for name in ("norm1", "norm2", "mlp"))
        assert not hasattr(tower, "merger")
        for x, grid, expected in calls:
            got = compiled(x, torch.tensor(grid))
            assert got.shape == expected.shape == (x.shape[0] // 4, config.text_config.hidden_size)
            torch.testing.assert_close(got.float(), expected.float(), atol=0.02, rtol=0.02)
