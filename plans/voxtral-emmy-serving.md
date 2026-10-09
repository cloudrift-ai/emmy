# Voxtral Mini 3B: next steps to Emmy serving

Voxtral Mini 3B (`mistralai/Voxtral-Mini-3B-2507`) serves with stock vLLM (the 1Cat sm_70 fork) on one Tesla
V100-SXM3-32GB, text and audio input. The Emmy golden `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` is
complete: nine traced programs cover the 30-layer Llama-architecture decoder, embeddings, final normalization and the
output head. Emmy serving is ineligible: the generation runner refuses any prompt with multimodal features. Seven of
the nine targets are also still slower than `torch.compile`. The recipe's `RESULTS.md` holds the measurements.

The goal: an Emmy serving image whose decoder runs Emmy kernels, with vLLM still running the Whisper-style audio encoder
and projector. It must transcribe as well as the stock lane and be at least as fast in time per output token.

## 1. Serve audio prompts through the Emmy runner

The runner already takes `inputs_embeds` in `forward`; what it lacks is the multimodal half of vLLM's model interface.

- Declare `SupportsMultiModal` on the Emmy generation model for a checkpoint with an audio tower. Delegate the audio
  encoder, the projector and the merge of audio embeddings into the token embeddings to vLLM's own Voxtral modules,
  loaded from the same checkpoint. Emmy owns only the decoder.
- Use the model's own multimodal processor and registry entry, so `input_audio` chat parts and
  `/v1/audio/transcriptions` reach the model unchanged.
- Voxtral uses plain RoPE, so the M-RoPE refusal does not apply. Keep that refusal for checkpoints whose vision input
  is still unserved.
- Loading: the stock recipe uses Mistral's format (`--config-format mistral --load-format mistral`). Emmy traces the
  HF `config.json`. Pick one format for both the encoder and the trunk, and check the tokenizer still matches.
- Verify on any CUDA card first: HF parity on an audio prompt (the transcript matches the stock engine word for word on
  a few LibriSpeech clips), then `emmy serve --runner generate` with an audio request.

Keep it model-agnostic: the same path should serve another audio-tower checkpoint with a supported decoder.

## 2. Close the V100 kernel losses

Measured at `-O3` (Emmy / `torch.compile`): pre-attention M1 2.70x, any width 2.65x, M8 1.69x; post-attention any
width 1.95x, M1 1.89x, M8 1.38x; output head 1.07x. Decode (M1) sets the time per output token, so start there.

- The Llama 3.1 8B V100 golden beats `torch.compile` on FP16 decode after #1078 (exact one-key softmax fold, vector
  weight reads). Check whether its decode rows apply to Voxtral's 3072-wide twins. If a schedule exists but is never
  offered, that is a corpus case. If it is offered but loses, it is a tuning or code generation gap.
- A `g8k` split piece failed with a misaligned address on this card, so the golden pins splits off. Reduce it to a
  realization case and fix it. Splits are the usual way decode GEMV fills 80 SMs.
- Tune the remaining losers one kernel at a time (`emmy run --golden … --kernel … --tune`). Re-bench the whole target
  and record only what wins in the layer.

## 3. Image, qualification and publication

- Publish the stock audio image `cloudriftai/1cat-vllm-sm70-audio:1.2.3-d76126608` (needs approval). Until then the
  recipe deploys only on a host that built it.
- Build the Emmy serving image on that audio base with the `release-serving-image` flow: golden coverage, HF parity,
  warm convergence, offline zero-recompile verification. Run the `EMMY_FAST_MATH=1` accuracy gate on transcription.
- A/B against the stock lane on the same V100 with the experiment's workloads, including the LibriSpeech
  transcription rows. Add a word error rate check to both lanes: today's rows measure speed only.

## Done when

The Voxtral recipe pins a published Emmy image. That image passes the audio deploy probe and transcription parity, and
it matches or beats the stock lane's time per output token at one stream on the V100.
