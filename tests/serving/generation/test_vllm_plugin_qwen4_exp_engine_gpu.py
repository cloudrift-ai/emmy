"""In-process vLLM engine serving a tiny Qwen3.8-Flash-Next through ``EmmyGenHybridModel``.

DELIBERATELY ``perf``-marked, for the reason ``test_vllm_plugin_gen_gpu`` gives: the in-process engine wants a
large free share of the card. One Gated DeltaNet layer, then one sparse-attention layer, both the 1Cat fork's,
between Emmy's gated-residual programs. vLLM keeps each request's Gated DeltaNet state in its KV cache blocks and
hands the fork's layer its metadata; Emmy's programs carry the four residual streams around it. The reference is
Hugging Face greedy generation on the same weights. Runs only in the pinned 1Cat image on sm_70, like
``test_vllm_plugin_qwen4_exp_gpu``.
"""

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

LAYER_TYPES = ["linear_attention", "full_attention"]
MAX_NEW = 8


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.qsa")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    import emmy.serving
    from tests.serving.helpers import save_tiny_qwen4_exp

    model_dir = tmp_path_factory.mktemp("tiny_qwen4_exp")
    save_tiny_qwen4_exp(model_dir, LAYER_TYPES)
    emmy.serving.register()
    with pytest.MonkeyPatch.context() as env:
        # The test process has CUDA initialized; vLLM's forked EngineCore would die on re-init.
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # The symbolic programs cover every width; the static tiers would only add compiles of unmeasured picks.
        env.setenv("EMMY_GEN_DECODE_BUCKET", "0")
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "0")
        llm = vllm.LLM(
            model=str(model_dir),
            runner="generate",
            hf_overrides={"architectures": ["EmmyGenHybridModel"]},
            enforce_eager=True,
            enable_prefix_caching=False,
            language_model_only=True,
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=256,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            gpu_memory_utilization=0.4,
        )
        engine = [llm, model_dir]
        yield engine
    engine.clear()
    del llm
    cleanup_dist_env_and_memory()


def _generate(llm, prompts):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, detokenize=False)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    return [list(out.outputs[0].token_ids) for out in outs]


def _reference(model_dir, prompt):
    import torch
    import transformers

    ref = transformers.Qwen4ExpForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        out = ref.generate(torch.tensor([prompt], device="cuda"), do_sample=False, max_new_tokens=MAX_NEW, use_cache=True)
    return out[0, len(prompt) :].tolist()


# 1 token: a request whose first step is a single token. 21 tokens: one prefill step. 100 tokens: above the 64 tokens
# a step holds, so the prompt spans two steps and the second continues a Gated DeltaNet state that is not zero.
PROMPTS = [[11], [7 + 13 * i for i in range(21)], [2 + 7 * i for i in range(100)]]


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: f"{len(p)}tok")
def test_hosted_mixers_match_hf_greedy(engine, prompt):
    """One request: token 0 comes from the prompt step, the rest from single-token steps that continue the state
    vLLM kept for the request."""
    llm, model_dir = engine
    assert _generate(llm, [prompt]) == [_reference(model_dir, prompt)]


def test_hosted_mixers_keep_concurrent_requests_apart(engine):
    """Several requests in one step: each must generate what it generates alone."""
    llm, model_dir = engine
    assert _generate(llm, PROMPTS) == [_reference(model_dir, prompt) for prompt in PROMPTS]
