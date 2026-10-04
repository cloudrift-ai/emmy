# Non-Interactive Model Qualification

The attached onboarding task is the complete run request. Its fields are the only source of the model, hardware,
credentials, and deadline; never select, substitute, or infer any of them, and never ask a question — a missing or
ambiguous field is an immediate failure. The attached copies of the `onboard-model` and
`run-experiment` skills are authoritative over any older copy in the checkout.

## Task fields

| Field | Use |
| --- | --- |
| `mode` | exactly `onboarding` or `verification`; selects the skill's recipe policy and is echoed in the summary |
| `model_id` | the exact Hugging Face model ID to qualify |
| `gpu` / `gpu_count` | the exact target platform; never change the GPU name, count, quantization, or checkpoint |
| `ssh_target`, `ssh_host`, `ssh_user`, `ssh_port` | the supplied server; the only host this run may use |
| `ssh_key` | pass `--ssh-key <value>` to every Emmy remote command |
| `deadline` | absolute wall-clock deadline for the whole run |
| `multimodal_mode` | `auto`, `multimodal`, or `text-only` qualification path |
| `publish_image` | `true` authorizes publishing a verified prebuilt Emmy image; `false` forbids it |
| `summary_path` | absolute path for the atomic summary, outside the repository |
| `expected_lifecycle` | in `verification` mode, the lifecycle tag the refreshed recipe must keep |

## Boundaries

Do not select a model or GPU, rent or delete the VM, commit, push, or touch a pull request; the caller owns those.
Tear down every deployed workload before returning. The caller owns the VM's lifetime, not its contents:
`ssh_user` has passwordless sudo, provisioning the node is your work under the skill's rule, and a host gate fails
only after Emmy's own provisioning ran and a specific step failed — quote that step's output in the summary.

For a recipe tagged `emmy-blocked`, inspect the Git history since its last onboarding report before repeating Emmy
work. Read changes relevant to the recorded blocker and verify the exact gate before clearing the tag. On success,
set `emmy.blocked` to match the recipe tag; an ineligible model without a concrete blocker uses `false` so nightly
selection may retry it. Keep a known blocker out of the high-priority Emmy queue without suppressing manual or
periodic verification.

For a missing image or an unfamiliar launch failure, check current official registries, release notes, engine
documentation, and upstream issues, then pin the exact working tag or digest. Delegate only bounded read-only research
or failure diagnosis to the `onboard-investigator` subagent, giving it the complete attached `investigate.md` and
exactly one question.

## Repository artifacts

Allowed areas: `recipes/` (including the model's `golden/`), `experiments/`, `docker/vllm-emmy-serve/models/`,
`tests/compiler/realization/cases/`, and the skill's bounded fixes under `emmy/` with their focused tests and nearest
`ARCHITECTURE.md`. A corpus case is evidence, not code: it does not count against the fix budget, and it is the focused
test for a compiler fix on its own.

Reuse the model's existing serving experiment root (many keep a platform-named root such as
`serving_v100_sxm2_16gb`) and create `experiments/<model>/serving/` only when there is none; never add a second root
for a platform an existing one already covers. The summary's `experiment` field is that root's `recipe.yaml` path,
never a directory. Git LFS is configured by the caller: verify the archive reports `filter: lfs`, but do not run
`git lfs track` or touch `.gitattributes`.

## Output

Always write the skill's summary to `summary_path`, on success and on failure, with `mode` set to the task's mode.
On failure it still names `recipes/<model>/RESULTS.md` with its dated failure entry, and lists the recipe with its
added `onboarding-failed` tag and the golden, corpus cases, and bounded fixes the run keeps; the caller commits them. The skill's summary
section owns every other part of the contract. Keep the failure message concise and credential-free; it goes to a chat
notification.
