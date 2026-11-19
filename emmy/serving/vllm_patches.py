"""Runtime workarounds for vLLM defects, applied from the plugin ``register`` hook.

Each patch documents the upstream defect, the upstream issue tracking it, and the pinned vLLM
version it targets; drop the patch when the pin moves to a release that closes that issue.
"""

import logging

logger = logging.getLogger(__name__)


def clamp_dummy_run_seq_lens() -> None:
    """vLLM 0.23: ``GPUModelRunner._dummy_run`` fills every dummy request's seq_len with the
    step's total TOKEN count. When a capture size exceeds ``--max-model-len`` (emmy's
    rider-top rung — chunk quantum + decode bucket — is the one rung that can), each dummy
    request claims a longer KV prefix than the block table can address (its rows hold
    ``cdiv(max_model_len, block_size)`` page ids). The unified-attention kernel loads block
    tables without a bounds mask, so the last request's programs read page ids from past the
    end of the tensor, and the garbage ids send the K/V loads far out of bounds — the
    capture-warmup illegal memory access. FLEX_ATTENTION trips over the same arithmetic
    loudly: its sliding-window block mask builds ``cdiv(seq_len, block_size)`` columns
    against a block table sliced to the row width. Head width is irrelevant — any model
    faults once a capture size passes max_model_len (verified by standalone kernel repro on
    RTX 4080: clean at 4096, faults at 4097, both head_dim 256 and 512).

    Clamping the seq-lens buffers to ``max_model_len`` before attention metadata is built
    restores the invariant every real batch already holds; legal batches are untouched.

    Only the DEFAULT model runner (``gpu_model_runner.py``) has this defect — and out-of-tree
    architectures like ``EmmyGenModel`` always get it (vLLM 0.23's newer runner is opt-in for
    stock Llama/Mistral/Qwen3 only, and its dummy batches use legal per-request seq lens).
    Verified end to end on the default runner: a 4128-token capture with max-model-len 4096
    warms up and captures cleanly with the clamp, greedy output correct.

    Reported upstream as vllm-project/vllm#53658 (with the standalone kernel repro); still open
    against vLLM main as of 2026-08-25. Drop this patch once the vllm pin reaches a release
    that closes it.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_emmy_seq_lens_clamped", False):
        return
    orig = GPUModelRunner._build_attention_metadata

    def build_with_clamped_seq_lens(self, *args, **kwargs):
        lens = getattr(self, "optimistic_seq_lens_cpu", None)
        if lens is None:
            return orig(self, *args, **kwargs)
        if int(lens.max()) > self.max_model_len:  # only dummy batches ever exceed
            lens.clamp_(max=self.max_model_len)
            # Same H2D refresh protocol as _dummy_run's own seq-lens fill.
            self.seq_lens.copy_(lens, non_blocking=True)
            if not getattr(self, "_emmy_seq_lens_clamp_logged", False):
                self._emmy_seq_lens_clamp_logged = True
                logger.info("emmy vllm patch: dummy-run seq lens clamped to max_model_len")
        return orig(self, *args, **kwargs)

    GPUModelRunner._build_attention_metadata = build_with_clamped_seq_lens
    GPUModelRunner._emmy_seq_lens_clamped = True


def one_token_prompt_rows_replay_decode_graphs() -> None:
    """vLLM's Model Runner V2 (the 1Cat fork's 1.5.1) refuses its whole-step decode graphs to any step that still holds
    a prompt token, so the last token of a prompt runs eager even when every row of the step is one token wide.
    Prefix caching ends a prompt that way: under ``--mamba-cache-mode align`` a 4K prompt runs as steps of 3,920, 175
    and 1 tokens. Eager, Emmy's runner calls its programs one at a time from the host, and on Qwen3.8-Flash-Next on four
    V100s that 1-token step took about 0.18 s against a decode step's 17 ms.

    The refusal guards speculative decoding: a prompt chunk of ``1 + k`` tokens has a verify step's shape but must start
    the recurrent state with prefill semantics. Without speculative decoding a decode row is one token, and the backends
    the graph holds already run a 1-token prompt row as a decode row: the Gated DeltaNet builder classes rows by query
    length unless speculative decoding is on, and the sparse attention has no separate prefill path. So with no
    speculative tokens, a step whose every row is one token is a uniform decode step whatever its phase, and replays the
    graph of its width. Drop this patch when the runner classes such a step as a decode itself."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:  # a vLLM without Model Runner V2
        return

    # vLLM 0.25.1, the RTX 5090 recipes' plugin lane, has Model Runner V2 but not this method.
    if getattr(GPUModelRunner, "_emmy_one_token_prompt_rows", False) or not hasattr(GPUModelRunner, "_get_uniform_decode_token_count"):
        return
    orig = GPUModelRunner._get_uniform_decode_token_count

    def uniform_decode_token_count(self, scheduler_output, dummy_run):
        if self.num_speculative_steps == 0 and set(scheduler_output.num_scheduled_tokens.values()) == {1}:
            return 1
        return orig(self, scheduler_output, dummy_run)

    GPUModelRunner._get_uniform_decode_token_count = uniform_decode_token_count
    GPUModelRunner._emmy_one_token_prompt_rows = True
