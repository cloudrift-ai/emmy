"""In-process vLLM engines serving a tiny Qwen3.8-Flash-Next: the 1Cat fork's own model, then ``EmmyGenHybridModel``.

DELIBERATELY ``perf``-marked, for the reason ``test_vllm_plugin_gen_gpu`` gives: an in-process engine wants a large
free share of the card. Two Gated DeltaNet layers, the second of which also reads the n-gram PLE, then one
sparse-attention layer. Emmy hosts the fork's token mixers between its own gated-residual programs and the PLE before
the second layer's, so it must generate exactly what the fork generates: the fork is the reference, not Hugging Face,
whose kernels pick other tokens on these random weights. vLLM keeps each request's Gated DeltaNet and short-convolution
state in its KV cache blocks and hands the fork's layers their metadata. Both engines run the fork's Model Runner V2,
the only one that lays out the sparse attention's ring cache. Runs only in the pinned 1Cat image on sm_70, like
``test_vllm_plugin_qwen4_exp_gpu``.

No prompt is a single token: the fork runs a one-token first step as a decode step, which reads the request's state
block without zeroing it first, so what it generates depends on what that block held.
"""

import contextlib

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

# The published order: layer 2 (one-indexed) reads the n-gram PLE, and the config allows it only on a Gated DeltaNet layer.
LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention"]
MAX_NEW = 8
# 2 tokens: one short prefill step. 21 tokens: one prefill step. 100 tokens: above the 64 tokens a step holds, so the
# prompt spans two steps and the second continues a Gated DeltaNet state that is not zero.
PROMPTS = [[3, 1], [7 + 13 * i for i in range(21)], [2 + 7 * i for i in range(100)]]
# Two prompts sharing 700 tokens, more than one KV cache block: the attention pages are widened to a Gated DeltaNet
# state's, about 400 tokens here, and a prefix cache keeps whole blocks only.
SHARED = [2 + (17 + 29 * i) % 1000 for i in range(700)]
PREFIXED = [SHARED + [3, 5, 7], SHARED + [11, 13]]
EMMY = {"hf_overrides": {"architectures": ["EmmyGenHybridModel"]}}
# Single-token decode steps captured, the width ``emmy serve`` records for an MoE model.
CAPTURED = {"enforce_eager": False, "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.qsa")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import emmy.serving
    from tests.serving.helpers import save_tiny_qwen4_exp

    model_dir = tmp_path_factory.mktemp("tiny_qwen4_exp")
    save_tiny_qwen4_exp(model_dir, LAYER_TYPES)
    emmy.serving.register()
    return vllm, model_dir


@contextlib.contextmanager
def _engine(vllm, model_dir, **extra):
    """One fresh engine, freed afterwards: vLLM freezes the objects it created at boot out of the garbage collector,
    and the next engine must not profile the card while this one's memory comes back."""
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    with pytest.MonkeyPatch.context() as env:
        # The test process has CUDA initialized; vLLM's forked EngineCore would die on re-init.
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # The symbolic programs cover every width. The static tiers would add programs no row measured for this
        # checkpoint, and a cold pick of the 16-token post program is one kernel that does not finish in a minute.
        env.setenv("EMMY_GEN_DECODE_BUCKET", "0")
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "0")
        # The fork picks its Model Runner V2 by architecture name, and Emmy's class is not one of its own.
        env.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        # Both engines read the n-gram table inside the worker. Booting the fork's own model would otherwise be free to
        # move it to a process of its own, and its defaults are written to this process's environment.
        env.setenv("VLLM_PLE_CPU_OFFLOAD", "0")
        # The table goes to host memory whole. Left to place itself, the fork sizes the KV cache for it before vLLM has
        # widened the attention pages to the Gated DeltaNet state, and refuses the narrower pages.
        env.setenv("VLLM_QWEN4EXP_PLE_HOST_GIB", "1")
        settings = dict(
            model=str(model_dir),
            runner="generate",
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
        llm = vllm.LLM(**(settings | extra))
        try:
            yield llm
        finally:
            del llm
            cleanup_dist_env_and_memory()


def _generate(llm, prompts):
    """Each request's greedy tokens and how many of its prompt tokens the prefix cache served."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, detokenize=False)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    return [(list(out.outputs[0].token_ids), out.num_cached_tokens or 0) for out in outs]


def _alone_then_together(llm):
    return [_generate(llm, [prompt])[0][0] for prompt in PROMPTS], [tokens for tokens, _ in _generate(llm, PROMPTS)]


@pytest.fixture(scope="module")
def served(checkpoint):
    vllm, model_dir = checkpoint
    with _engine(vllm, model_dir) as llm:
        fork = _alone_then_together(llm)
    with _engine(vllm, model_dir, **EMMY) as llm:
        emmy = _alone_then_together(llm)
    return fork, emmy


@pytest.mark.parametrize("index", range(len(PROMPTS)), ids=[f"{len(p)}tok" for p in PROMPTS])
def test_hosted_mixers_generate_what_the_fork_generates(served, index):
    """One request: token 0 comes from the prompt steps, the rest from single-token steps that continue the state
    vLLM kept for the request."""
    (fork, _), (emmy, _) = served
    assert emmy[index] == fork[index]


def test_hosted_mixers_keep_concurrent_requests_apart(served):
    """Several requests in one step, each on its own state."""
    (_, fork), (_, emmy) = served
    assert emmy == fork


@pytest.fixture(scope="module")
def served_as_deployed(checkpoint):
    """The fork's own model, then Emmy capturing its decode steps, both with prefix caching on: each short prompt alone
    and then together, then the two prompts that share a prefix, one after the other. One capturing engine per
    process: a second one's capture trips an internal assert in PyTorch's CUDA allocator."""
    vllm, model_dir = checkpoint
    # A step holds a whole block at least: vLLM keeps the state only at block ends.
    caching = {"enable_prefix_caching": True, "mamba_cache_mode": "align", "max_model_len": 1024, "max_num_batched_tokens": 512}
    runs = []
    for extra in ({}, CAPTURED | EMMY):
        with _engine(vllm, model_dir, **caching, **extra) as llm:
            runs.append((*_alone_then_together(llm), [_generate(llm, [prompt])[0] for prompt in PREFIXED]))
    return runs


def test_captured_decode_generates_what_the_fork_generates(served_as_deployed):
    """Emmy's decode steps under CUDA graph capture, at the one width ``emmy serve`` records for an MoE model: a
    single-token step replays its graph, the symbolic programs and the fixed-slot experts with the hosted layers, their
    state and the PLE's inputs at the addresses it was recorded with. A wider step runs eager."""
    (fork_alone, fork_together, _), (alone, together, _) = served_as_deployed
    assert alone == fork_alone
    assert together == fork_together


def test_a_prefix_cache_hit_generates_what_the_fork_generates(served_as_deployed):
    """The second prompt resumes from the blocks the first one cached: the attention pages, and the Gated DeltaNet
    and short-convolution states vLLM keeps at a block's end (``mamba_cache_mode`` "align"). Emmy must serve both
    requests as the fork does, from the same cached tokens."""
    (*_, fork), (*_, emmy) = served_as_deployed
    assert fork[1][1] > 0, "the second prompt did not hit the prefix cache"
    assert emmy == fork
