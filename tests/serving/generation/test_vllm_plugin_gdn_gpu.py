"""In-process vLLM engine test of GDN (gated DeltaNet) serving.

DELIBERATELY ``perf``-marked, for the reason ``test_vllm_plugin_gen_gpu`` gives: the in-process engine wants a
large free share of the card. Needs CUDA + vllm. Saves a tiny random hybrid Qwen3.5 — one GDN layer, then one
full-attention layer with an output gate — and serves it through ``EmmyGenHybridModel``. vLLM keeps each request's
GDN state in its KV cache blocks; the model class finds the state per request and the runner computes the layer.
The reference is Hugging Face greedy generation on the same weights.

The GDN kernels take their schedules from the serving tests' golden: an uncut GDN kernel takes minutes per call.
The attention programs, which that golden does not hold in float16, compile cold.
"""

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # cached; vocab 32000
MAX_NEW = 8


def _save_tiny_hybrid(path):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from tests.compiler.trace.test_huggingface import _QWEN3_5_TINY

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(**(_QWEN3_5_TINY | {"vocab_size": 32000, "max_position_embeddings": 512}))
    Qwen3_5ForCausalLM(config).eval().to(torch.float16).save_pretrained(path)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, with the tiny hybrid checkpoint it serves."""
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("transformers.models.qwen3_5")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import emmy.serving
    from emmy.compiler.pipeline.search.golden import evidence_scope
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.serving import helpers

    model_dir = tmp_path_factory.mktemp("tiny_qwen3_5_hybrid")
    _save_tiny_hybrid(str(model_dir))
    emmy.serving.register()
    document = helpers.golden_document()
    with pytest.MonkeyPatch.context() as env:
        # The test process has CUDA initialized; vLLM's forked EngineCore would die on re-init.
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # The widths the golden's GDN rows cover: 16, 4 and 1.
        env.setenv("EMMY_GEN_DECODE_BUCKET", "4")
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "16")
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
            )
        yield llm, model_dir


def _generate(llm, prompts):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(temperature=0.0, max_tokens=MAX_NEW))
    return [list(out.outputs[0].token_ids) for out in outs]


def _reference(model_dir, prompt):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    ref = Qwen3_5ForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        out = ref.generate(torch.tensor([prompt], device="cuda"), do_sample=False, max_new_tokens=MAX_NEW, use_cache=True)
    return out[0, len(prompt) :].tolist()


# 1 token: the request's first step is a single token, the case where a stale KV cache block would show.
# 5 and 21 tokens: GDN widths 4 + 1 and 16 + 4 + 1.
PROMPTS = [[11], [3, 1, 4, 1, 5], [7 + 13 * i for i in range(21)]]


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: f"{len(p)}tok")
def test_gdn_engine_matches_hf_greedy(engine, prompt):
    """One request: token 0 comes from the prompt step, the rest from single-token steps that continue the state
    vLLM kept in the request's KV cache block."""
    llm, model_dir = engine
    assert _generate(llm, [prompt]) == [_reference(model_dir, prompt)]


def test_gdn_engine_keeps_concurrent_requests_apart(engine):
    """Several requests in one step: each must generate what it generates alone. A step packs their tokens back to
    back, and each GDN layer runs them one request after another, each on its own state."""
    llm, model_dir = engine
    assert _generate(llm, PROMPTS) == [_reference(model_dir, prompt) for prompt in PROMPTS]
    # A later request reuses KV cache blocks that earlier ones freed: its state must start from zero again.
    assert _generate(llm, PROMPTS[::-1]) == [_reference(model_dir, prompt) for prompt in PROMPTS[::-1]]
