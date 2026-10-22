"""In-process vLLM engine test of an Emmy-served MTP head drafting for a GDN model.

``perf``-marked for the reason ``test_vllm_plugin_gdn_gpu`` gives, and serving the same tiny random hybrid Qwen3.5
with a random one-layer MTP head added to its checkpoint. vLLM's MTP proposer drafts through ``EmmyMTPModel``,
whose programs Emmy compiles from the head's checkpoint weights. A random head's drafts are rarely accepted, so this
checks that the head drafts and that the target's verify steps keep greedy's tokens; the head's numerics are
checked on a real checkpoint, by its acceptance rate.
"""

import pytest

from tests.serving.generation.test_vllm_plugin_gdn_gpu import TOKENIZER, _save_tiny_hybrid

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

MAX_NEW = 16
DRAFTS = 2
PROMPTS = [[5, 6, 7] * 4, [11]]


def _add_mtp_head(path):
    """Write a random one-layer MTP head into the checkpoint, under the names a Qwen3.5 checkpoint gives it."""
    import json

    import torch
    from safetensors.torch import load_file, save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config_path = path / "config.json"
    raw = json.loads(config_path.read_text())
    config = Qwen3_5TextConfig(**(raw | {"num_hidden_layers": 1, "layer_types": ["full_attention"]}))
    torch.manual_seed(1)
    head = Qwen3_5TextModel(config).to(torch.float16)
    hidden = config.hidden_size
    tensors = {f"mtp.layers.0.{name}": value for name, value in head.layers[0].state_dict().items()}
    tensors["mtp.norm.weight"] = torch.randn(hidden, dtype=torch.float16) * 0.1
    tensors["mtp.pre_fc_norm_embedding.weight"] = torch.randn(hidden, dtype=torch.float16) * 0.1
    tensors["mtp.pre_fc_norm_hidden.weight"] = torch.randn(hidden, dtype=torch.float16) * 0.1
    tensors["mtp.fc.weight"] = torch.randn(hidden, 2 * hidden, dtype=torch.float16) * (2 * hidden) ** -0.5
    weights = path / "model.safetensors"
    save_file(load_file(weights) | {name: value.contiguous() for name, value in tensors.items()}, weights, metadata={"format": "pt"})
    config_path.write_text(json.dumps(raw | {"mtp_num_hidden_layers": 1}))


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("transformers.models.qwen3_5")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import emmy.serving
    from emmy.compiler.pipeline.search.golden import evidence_scope
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.serving import helpers

    helpers.skip_without_gdn_verify_counts()
    model_dir = tmp_path_factory.mktemp("tiny_qwen3_5_hybrid_mtp")
    _save_tiny_hybrid(str(model_dir))
    _add_mtp_head(model_dir)
    emmy.serving.register()
    document = helpers.golden_document()
    with pytest.MonkeyPatch.context() as env:
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # The model runner the serving image's vLLM runs: it hands a GDN layer each request's accepted count.
        env.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        env.setenv("EMMY_GEN_DECODE_BUCKET", "4")
        # Steps of up to 4 rows: the GDN widths 4 and 1 and the verify widths.
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "4")
        with pinned_knobs(document.shared_regime()), evidence_scope([document]):
            llm = vllm.LLM(
                model=str(model_dir),
                tokenizer=TOKENIZER,
                runner="generate",
                hf_overrides={"architectures": ["EmmyGenHybridModel"]},
                enforce_eager=True,
                dtype="float16",
                max_model_len=128,
                max_num_batched_tokens=64,
                max_num_seqs=4,
                gpu_memory_utilization=0.4,
                enable_prefix_caching=False,
                disable_log_stats=False,
                speculative_config={"method": "mtp", "num_speculative_tokens": DRAFTS},
            )
        yield llm, model_dir


def test_emmy_mtp_head_drafts_and_keeps_greedy_tokens(engine):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from emmy.serving.vllm_model_mtp import EmmyMTPModel
    from tests.serving.generation.test_vllm_plugin_gdn_spec_gpu import _reference

    llm, model_dir = engine
    drafter = llm.llm_engine.model_executor.driver_worker.worker.model_runner.drafter
    assert isinstance(drafter.model, EmmyMTPModel)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in PROMPTS], params)
    expected = [_reference(model_dir, prompt)[0][:MAX_NEW] for prompt in PROMPTS]
    assert [list(out.outputs[0].token_ids) for out in outs] == expected
    assert sum(m.value for m in llm.get_metrics() if m.name == "vllm:spec_decode_num_drafts") > 0
