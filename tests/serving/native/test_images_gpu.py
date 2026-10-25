"""An image prompt through native generation on a tiny Qwen3.5 vision-language checkpoint: the tower's programs, the
merge of its rows into the prompt's embeddings and the M-RoPE rotary give transformers' logits."""

import asyncio
import json
import shutil

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.native import NativeWorker
from emmy.serving.native.prepare import export_model, load_model
from tests.compiler.helpers import requires_cuda
from tests.serving.generation.test_vision_tower import VISION_TINY
from tests.serving.helpers import QWEN3_5_TINY

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]

#: Token IDs inside the tiny vocabulary for the image's placeholders.
IMAGE, VIDEO, START, END = 60, 61, 58, 59


def _checkpoint(root):
    """A random tiny Qwen3.5 vision-language checkpoint in FP16, written to ``root``: two full-attention layers, so
    the prompt prefills in chunks, and a whole-head M-RoPE whose three sections all rotate."""
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    rope = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    }
    text = QWEN3_5_TINY | {"layer_types": ["full_attention", "full_attention"], "rope_parameters": rope}
    config = Qwen3_5Config(
        text_config=text,
        vision_config=VISION_TINY,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=START,
        vision_end_token_id=END,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).half().eval()
    model.save_pretrained(root)
    model.config._attn_implementation = "eager"
    model.config.vision_config._attn_implementation = "eager"
    return model


@pytest.mark.parametrize("vision_gpu", [None, 1], ids=["resident", "paged"])
def test_an_image_prompt_gives_transformers_logits(tmp_path, monkeypatch, vision_gpu):
    """The tower runs at the narrow width over a padded 8 x 12 patch grid (24 image tokens), the trunk merges its rows
    in place of the image tokens through chunked prefill and the one-token tail, and every logit from the prompt's last
    token through three decode steps matches transformers' vision-language model at the M-RoPE positions it computes;
    a text request on the same pack matches too. With one block of the tower's two on the device, the other moves
    through the pager's slot, and every program runs captured."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    if vision_gpu is None:
        monkeypatch.delenv("EMMY_GEN_VISION_GPU", raising=False)
    else:
        monkeypatch.setenv("EMMY_GEN_VISION_GPU", str(vision_gpu))
    capture = vision_gpu is not None
    model = _checkpoint(tmp_path / "checkpoint")
    lm, ckpt = load_model(str(tmp_path / "checkpoint"))
    assert ckpt is None and lm.dtype == torch.float16
    with gpu_lock():
        root = export_model(lm, tmp_path / "pack", context_length=64, prefill_size=8, vision=str(tmp_path / "checkpoint"))
    vision = json.loads((root / "manifest.json").read_text())["key"]["generation"]["vision"]
    assert vision["image_token_id"] == IMAGE and vision["widths"] == [256, 1024]
    # Paged, every projection weight of a block is one page of a pool, each pool a page per block.
    assert (len(vision["pools"]), {len(blocks) for blocks in vision["pools"].values()}) == ((4, {2}) if vision_gpu else (0, set()))

    grid = torch.tensor([[1, 8, 12]])
    patches, tokens = 96, 24
    rows = 4 * vision["widths"][0]
    torch.manual_seed(1)
    pixels = torch.randn(patches, 3 * 2 * 8 * 8).half()
    prompt = [5, 6, START, *([IMAGE] * tokens), END, 7, 8, 9]
    kinds = torch.tensor([[int(t == IMAGE) for t in prompt]])
    positions, _ = model.model.get_rope_index(torch.tensor([prompt]), kinds, grid)

    def reference(prefix, pixels=pixels):
        ids = torch.tensor([prefix])
        image = {"pixel_values": pixels, "image_grid_thw": grid} if IMAGE in prefix else {}
        kind = torch.tensor([[int(t == IMAGE) for t in prefix]])
        with torch.no_grad():
            return model(input_ids=ids, mm_token_type_ids=kind, **image).logits[0, -1].float().numpy()

    # The image moves the logits well past the tolerance below, so a merge that dropped it would fail.
    assert np.abs(reference(prompt) - reference(prompt, torch.zeros_like(pixels))).max() > 0.2

    def padded(values, width, axis=0):
        shape = list(values.shape)
        shape[axis] = width - shape[axis]
        return np.concatenate([values, np.zeros(shape, values.dtype)], axis=axis)

    index, weight = get_vision_interpolation_indices_and_weights(grid, 4, mode="bilinear", align_corners=True, spatial_merge_size=2)
    inputs = {
        "patches": padded(pixels.numpy(), rows),
        "pos_index": padded(index.T.numpy().astype(np.int64), rows, axis=1),
        "pos_weight": padded(weight.T.half().numpy(), rows, axis=1),
        "positions": padded(get_vision_position_ids(grid, 2).numpy().astype(np.int64), rows),
        "patches_valid": np.array([patches], np.int64),
    }
    files = {}
    for name, values in inputs.items():
        files[name] = str(tmp_path / f"{name}.bin")
        values.tofile(files[name])
    path, positions_path, logits_path = tmp_path / "prompt.bin", tmp_path / "mrope.bin", tmp_path / "logits.bin"
    np.asarray(prompt, np.int64).tofile(path)
    positions[:, 0].T.numpy().astype(np.int64).tofile(positions_path)
    monkeypatch.setenv("PATH", "/nonexistent")

    async def run(image):
        worker = NativeWorker(executable=executable)
        try:
            await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=120)
            text = prompt if image else [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            np.asarray(text, np.int64).tofile(path)
            start = {"op": "start_generation", "prompt": str(path)}
            if image:
                start["image"] = {"tokens": tokens, "inputs": files, "positions": str(positions_path), "capture": capture}
            await worker.run_job(start, wall_timeout_s=120)
            prefix = list(text)
            for _ in range(4 + len(text)):
                result = await worker.run_job(
                    {"op": "generation_step", "capture": capture, "prefill": True, "logits": str(logits_path)}, wall_timeout_s=120
                )
                if result["token"] is None:
                    continue
                actual = np.fromfile(logits_path, np.float32)
                np.testing.assert_allclose(actual, reference(prefix), rtol=2e-2, atol=2e-2)
                prefix.append(result["token"])
                if len(prefix) == len(text) + 4:
                    break
            assert len(prefix) == len(text) + 4
        finally:
            await worker.aclose()

    with gpu_lock():
        asyncio.run(run(image=True))
        asyncio.run(run(image=False))
