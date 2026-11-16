# emmy-native — a recipe's native lane as one serving image

A recipe's native lane (`engine.llm.native`, the [recipe reference](../../emmy/recipe/ARCHITECTURE.md)) deploys as one
image: Emmy's native server, `emmy-server`, its runtime worker, and the pack prepared from the recipe golden. The pack
holds every compiled program, the weights in the form the runtime reads, the tokenizer, the chat template and, for a
pack that answers images, the image processor config ([native generation](../../emmy/serving/native/ARCHITECTURE.md)).
A boot loads it and compiles nothing: the image has no Python, no compiler and no CUDA toolkit, only the driver's
`libcuda`, which the NVIDIA container runtime mounts. The image's entrypoint is the server over its pack;
`emmy deploy` passes the served model ID, the address and the recipe's context.

```text
cloudriftai/emmy-native-<model-slug>:<emmy version>-<source sha>
```

The slug is `emmy.publish.model_slug`, as for the vLLM images; the version is the Emmy package's and the sha the
checkout's. `emmy publish RECIPE --engine native` checks the labels the bake writes (family `emmy-native`, version,
revision, model ID) against the recipe's image reference and pushes it, after the same separate human approval as
any other image.

## The bake

`make native-image RECIPE=recipes/<model>` runs `bake.sh` on the target card:

1. **Prepare the pack.** `emmy.serving.native.image` resolves the recipe's native lane for the card and prints the
   preparation: `emmy serve MODEL --runner generate --native --export DIR` at the recipe's revision and context, from
   the lane's golden alone under strict evidence, with the lane's flags (`extra_args`: the KV cache dtype, the page
   size, images, the drafter) and environment (`extra_env`: the widths and the pager budgets). The flags and budgets
   are fixed in the pack, so they belong to the bake, not to the server. With `PREPARE_IMAGE` naming an Emmy serving
   image built from the same checkout (`make vllm-emmy-image`), the preparation runs in it, so the pack's cubins come
   from that image's nvcc, the toolchain the golden rows were recorded with.
2. **Build the binaries** from the same checkout (`cargo build --release --locked`). They are built on the bake host,
   so the base image's glibc must be at least the host's.
3. **Split and build.** `split_hf.sh` (shared with [vllm-emmy-serve](../vllm-emmy-serve/ARCHITECTURE.md)) balances
   the pack into 24 hardlinked sub-10 GB parts, since Docker Hub refuses larger blobs, and the Dockerfile copies them
   back into one directory.

Nothing is pushed. The recipe then names the tag the bake printed, in a later commit than the one baked, as the vLLM
images' recipes do.
