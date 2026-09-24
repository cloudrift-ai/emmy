"""Opt-in checkpoint text parity against Transformers, including native Jinja rendering."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_checkpoint_text_parity(request, tmp_path):
    checkpoint = request.config.getoption("--native-checkpoint")
    if not checkpoint or not shutil.which("cargo"):
        pytest.skip("requires --native-checkpoint and Cargo")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "chat_template.jinja").write_text(tokenizer.chat_template)
    cases = []
    for prompt in ["Hello world", "你好 🦀 café", " leading\nspaces\t", "<|im_start|>user\nhello", "�"]:
        ids = tokenizer.encode(prompt)
        cases.append({"prompt": prompt, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=True)})
    for messages in [
        [{"role": "user", "content": "Hello 🦀"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "你好"}],
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "<think>reason</think>\nanswer"},
         {"role": "user", "content": "two"}],
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "answer"}],
    ]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        cases.append({"messages": messages, "prompt": prompt, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=True)})
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps({"root": str(tmp_path), "cases": cases}))
    root = Path(__file__).resolve().parents[3]
    subprocess.run(["cargo", "test", "--locked", "-p", "emmy-server", "--test", "text_parity"], cwd=root,
                   env={**os.environ, "NATIVE_TEXT_FIXTURE": str(fixture)}, check=True, timeout=90)
