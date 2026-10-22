"""In-process vLLM engine test of speculative decoding on GDN (gated DeltaNet) layers.

``perf``-marked for the reason ``test_vllm_plugin_gdn_gpu`` gives, and serving a tiny random hybrid Qwen3.5 like it.
Draft tokens come from n-gram lookup. A random model never repeats itself, so this one ties its embedding to its head
and damps every layer's output: it mostly repeats its last token, and the drafts are accepted. A verify step then
keeps the GDN state after each of its rows, and the next step continues from the state after its last accepted
token. The reference is Hugging Face greedy decoding on the same weights, without drafts: the tokens, and each token's
log-probability, which a wrong state moves even where the greedy token stays.
"""

import pytest

from tests.serving.generation.test_vllm_plugin_gdn_gpu import TOKENIZER

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

MAX_NEW = 24
DRAFTS = 2
# Repeating prompts, whose continuations an n-gram lookup drafts from.
PROMPTS = [[5, 6, 7] * 4, [3, 1, 4, 1, 5, 9, 2, 6] * 2, [11]]


def _save_repeating_hybrid(path):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from tests.serving.helpers import QWEN3_5_TINY

    torch.manual_seed(0)
    rope = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2], "mrope_interleaved": True}
    tiny = QWEN3_5_TINY | {"vocab_size": 32000, "max_position_embeddings": 512, "head_dim": 64, "rope_parameters": rope}
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tiny, tie_word_embeddings=True)).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight")):
                param.mul_(0.2)
    model.to(torch.float16).save_pretrained(path)


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
    model_dir = tmp_path_factory.mktemp("tiny_qwen3_5_hybrid_spec")
    _save_repeating_hybrid(str(model_dir))
    emmy.serving.register()
    document = helpers.golden_document()
    with pytest.MonkeyPatch.context() as env:
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
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
                speculative_config={
                    "method": "ngram_gpu",
                    "num_speculative_tokens": DRAFTS,
                    "prompt_lookup_min": 1,
                    "prompt_lookup_max": 3,
                },
            )
        yield llm, model_dir


def _reference(model_dir, prompt):
    """Greedy decoding with the whole sequence recomputed each step: ``(tokens, log-probabilities)``."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    ref = Qwen3_5ForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).to("cuda").eval()
    tokens, logprobs = list(prompt), []
    with torch.no_grad():
        for _ in range(MAX_NEW):
            row = torch.log_softmax(ref(torch.tensor([tokens], device="cuda")).logits[0, -1].float(), dim=-1)
            tokens.append(int(row.argmax()))
            logprobs.append(float(row[tokens[-1]]))
    return tokens[len(prompt) :], logprobs


def _counter(llm, name):
    return sum(m.value for m in llm.get_metrics() if m.name == f"vllm:spec_decode_{name}")


def test_gdn_verify_steps_match_hf_greedy(engine):
    """Requests alone, then together: the tokens are greedy's, and drafts were accepted, so some steps continued
    from a state that a verify step kept past its first row."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm, model_dir = engine
    expected = [_reference(model_dir, prompt) for prompt in PROMPTS]
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, ignore_eos=True, logprobs=0)

    def generate(prompts):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
        return [list(out.outputs[0].token_ids) for out in outs], [
            [next(iter(step.values())).logprob for step in out.outputs[0].logprobs] for out in outs
        ]

    for together in ([prompt] for prompt in PROMPTS), [PROMPTS]:
        for prompts in together:
            tokens, logprobs = generate(prompts)
            want = [expected[PROMPTS.index(prompt)] for prompt in prompts]
            assert tokens == [tokens for tokens, _ in want]
            for got, (_, ref) in zip(logprobs, want, strict=True):
                assert got == pytest.approx(ref, abs=0.05)
    assert _counter(llm, "num_accepted_tokens") > 0, _counter(llm, "num_draft_tokens")
