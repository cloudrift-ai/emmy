# Qwen3-Embedding-0.6B discovery

## Assessment on 2026-10-05

Maintained, heat 72. Earlier note stands: only embedding repo with current momentum this period and strong HF trending
signal. This cycle's raw download leaders on embeddings are the older standards (BAAI/bge-m3 at 34,127,223 30d;
nomic-ai/nomic-embed-text-v1.5 at 13,054,312 30d), which is expected for production pipelines and does not displace a
fresh small embedding release from the current-momentum slot. Consumes one 5090 or 4090 at 4K context.

## Assessment on 2026-10-17

Maintained, heat 45 (lowered from the 72 recorded at the 2026-10-05 write — named observation: the Qwen embedding
line no longer holds the current-momentum slot on its own. The 2026-10-17 capture shows the #1 trending repository
platform-wide is google/embeddinggemma-2 (trendingScore 921, ~2,533 likes, ~151K 30-day downloads at
https://huggingface.co/google/embeddinggemma-2), and the Qwen3-Embedding-0.6B node did not resurface as a top-trending
embedding this cycle, so the momentum read is no longer "strong fresh trending" but "stable, established small
embedding with solid serving value." The recipe stays maintained for that serving slot — a small, stable, runnable
embedding on one 5090 or 4090 — but the heat is onboarding priority, not quality; it no longer leads the embedding
wave embeddinggemma-2 now does.
