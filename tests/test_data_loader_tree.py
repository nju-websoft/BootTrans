# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import types

import numpy as np
import torch
from pathlib import Path

from verl.trainer.config import DataConfig
from verl.trainer.data_loader_tree import create_tree_dataloader


class DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, tokenize: bool = False):
        content = messages[0]["content"]
        return content if isinstance(content, str) else str(content)

    def __call__(self, texts, add_special_tokens: bool = False, return_tensors: str = "pt"):
        token_seqs = []
        mask_seqs = []
        max_len = 0
        for text in texts:
            tokens = list(range(1, max(len(text.split()), 1) + 1))
            max_len = max(max_len, len(tokens))
            token_seqs.append(torch.tensor(tokens, dtype=torch.long))
        for tokens in token_seqs:
            pad_len = max_len - tokens.numel()
            if pad_len > 0:
                padding = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=-1)
            mask = torch.ones_like(tokens)
            mask_seqs.append(mask)
        return {
            "input_ids": torch.stack(token_seqs, dim=0),
            "attention_mask": torch.stack(mask_seqs, dim=0),
        }

    def encode(self, text, add_special_tokens: bool = False):
        return list(range(1, max(len(text.split()), 1) + 1))


def _build_config(train_path: str) -> DataConfig:
    return DataConfig(
        train_files=train_path,
        val_files=train_path,
        prompt_key="source_prompt",
        answer_key="python_test",
        max_prompt_length=16,
        rollout_batch_size=2,
        mini_rollout_batch_size=2,
        format_prompt=None,
        shuffle=False,
        filter_overlong_prompts=False,
        val_batch_size=2,
    )


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_tree_dataloader_priority(tmp_path):
    seed_records = [
        {"question_id": "seed1", "source_prompt": "s1", "python_test": "assert True"},
        {"question_id": "seed2", "source_prompt": "s2", "python_test": "assert True"},
        {"question_id": "seed3", "source_prompt": "s3", "python_test": "assert True"},
    ]
    seed_file = tmp_path / "seed.jsonl"
    _write_jsonl(seed_file, seed_records)

    exploration_records = [
        {"question_id": "exp1", "source_prompt": "e1", "python_test": "assert True"},
        {"question_id": "exp2", "source_prompt": "e2", "python_test": "assert True"},
        {"question_id": "exp3", "source_prompt": "e3", "python_test": "assert True"},
    ]
    explore_file = tmp_path / "explore.jsonl"
    _write_jsonl(explore_file, exploration_records)

    config = _build_config(str(seed_file))
    tokenizer = DummyTokenizer()
    train_loader, val_loader = create_tree_dataloader(config, tokenizer, processor=None)
    train_loader.add_exploration_jsonl(str(explore_file))

    iterator = iter(train_loader)
    batch1 = next(iterator)
    batch2 = next(iterator)

    assert batch1["question_id"].tolist() == ["exp1", "exp2"]
    assert batch2["question_id"].tolist() == ["exp3", "seed1"]

    val_batch = next(iter(val_loader))
    assert val_batch["input_ids"].shape[0] == 2
    assert set(val_batch["question_id"].tolist()).issuperset({"seed1", "seed2", "seed3"})


def test_tree_dataloader_complex_mix(tmp_path):
    seed_records = [
        {"question_id": "seed1", "source_prompt": "s1", "python_test": "assert True"},
        {"question_id": "seed2", "source_prompt": "s2", "python_test": "assert True"},
        {"question_id": "seed3", "source_prompt": "s3", "python_test": "assert True"},
        {"question_id": "seed4", "source_prompt": "s4", "python_test": "assert True"},
    ]
    seed_file = tmp_path / "seed.jsonl"
    _write_jsonl(seed_file, seed_records)

    config = _build_config(str(seed_file))
    tokenizer = DummyTokenizer()
    train_loader, _ = create_tree_dataloader(config, tokenizer, processor=None)

    # initial exploration (spans >1 batch)
    explore1 = tmp_path / "explore1.jsonl"
    _write_jsonl(
        explore1,
        [
            {"question_id": "exp1", "source_prompt": "e1", "python_test": "assert True"},
            {"question_id": "exp2", "source_prompt": "e2", "python_test": "assert True"},
            {"question_id": "exp3", "source_prompt": "e3", "python_test": "assert True"},
        ],
    )
    train_loader.add_exploration_jsonl(str(explore1))

    iterator = iter(train_loader)
    batch1 = next(iterator)  # exp1, exp2

    # add more exploration mid-iteration
    explore2 = tmp_path / "explore2.jsonl"
    _write_jsonl(
        explore2,
        [
            {"question_id": "exp4", "source_prompt": "e4", "python_test": "assert True"},
            {"question_id": "exp5", "source_prompt": "e5", "python_test": "assert True"},
        ],
    )
    train_loader.add_exploration_jsonl(str(explore2))

    batch2 = next(iterator)  # exp3, exp4
    batch3 = next(iterator)  # exp5, seed1
    batch4 = next(iterator)  # seed2, seed3
    batch5 = next(iterator)  # seed4, seed1 (wrap)

    assert batch1["question_id"].tolist() == ["exp1", "exp2"]
    assert batch2["question_id"].tolist() == ["exp3", "exp4"]
    assert batch3["question_id"].tolist() == ["exp5", "seed1"]
    assert batch4["question_id"].tolist() == ["seed2", "seed3"]
    assert batch5["question_id"].tolist() == ["seed4", "seed1"]


if __name__ == "__main__":
    test_tree_dataloader_complex_mix(Path("./examples"))
