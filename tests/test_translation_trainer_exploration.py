# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from types import SimpleNamespace

from verl.protocol import DataProto
from verl.trainer.ray_translation_trainer import RayCodeTranslationPPOTrainer
from transformers import AutoTokenizer


class DummyTrainer(RayCodeTranslationPPOTrainer):
    def __init__(self):
        # bypass base init
        pass


def test_maybe_add_to_exploration():
    trainer = DummyTrainer()
    trainer.config = SimpleNamespace(data=SimpleNamespace(prompt_key="source_prompt", answer_key="python_test"))
    exploration_pool = []

    def add_exploration_data(examples):
        exploration_pool.extend(examples)

    trainer.train_dataloader = SimpleNamespace(add_exploration_data=add_exploration_data)

    batch = DataProto(
        batch=None,
        non_tensor_batch={
            "uid": np.array(["u1", "u2"], dtype=object),
            "ground_truth": np.array(["gt1", "gt2"], dtype=object),
            "source_prompt": np.array(["code1", "code2"], dtype=object),
            "depth": np.array([2, 3], dtype=object),
        },
    )

    trainer._maybe_add_to_exploration(batch, indices=[0, 1])

    assert len(exploration_pool) == 2
    assert exploration_pool[0]["source_prompt"] == "code1"
    assert exploration_pool[0]["python_test"] == "gt1"
    assert exploration_pool[0]["depth"] == 2
    assert exploration_pool[1]["source_prompt"] == "code2"
    assert exploration_pool[1]["python_test"] == "gt2"
    assert exploration_pool[1]["depth"] == 3


def test_online_filter_adds_exploration(monkeypatch, tmp_path, tokenizer_path='/data1/shares/Qwen2.5-7B-Instruct'):
    # Prepare trainer with minimal stubs
    trainer = DummyTrainer()
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(
            prompt_key="source_prompt",
            answer_key="python_test",
            min_pixels=None,
            max_pixels=None,
            video_fps=1.0,
            max_prompt_length=8,
            format_prompt=str(tmp_path / "tpl.jinja"),
        ),
        algorithm=SimpleNamespace(filter_key="overall", online_filtering=True, adv_estimator="grpo"),
        worker=SimpleNamespace(rollout=SimpleNamespace(n=1)),
        trainer=SimpleNamespace(max_try_make_batch=1),
    )
    exploration_pool = []

    def add_exploration_data(examples):
        exploration_pool.extend(examples)

    trainer.train_dataloader = SimpleNamespace(add_exploration_data=add_exploration_data)
    tpl_path = tmp_path / "tpl.jinja"
    tpl_path.write_text("{{source_prompt}} {{target_signature}}")
    trainer.config.data.format_prompt = str(tpl_path)

    import ray

    monkeypatch.setattr(ray, "get", lambda x: x)

    def generate_sequences(prompts):
        bs = len(prompts)
        responses = torch.ones((bs, 1), dtype=torch.long)
        response_mask = torch.ones_like(responses)
        batch = {
            "input_ids": prompts.batch["input_ids"],
            "attention_mask": prompts.batch["attention_mask"],
            "position_ids": prompts.batch["position_ids"],
            "responses": responses,
            "response_mask": response_mask,
            "ref_log_probs": torch.zeros_like(responses, dtype=torch.float32),
            "old_log_probs": torch.zeros_like(responses, dtype=torch.float32),
        }
        return DataProto.from_dict(tensors=batch, non_tensors={}, meta_info=prompts.meta_info)

    trainer.actor_rollout_ref_wg = SimpleNamespace(generate_sequences=generate_sequences)
    trainer.reward_fn = SimpleNamespace(
        compute_reward=SimpleNamespace(
            remote=lambda data: (
                torch.ones_like(data.batch["responses"], dtype=torch.float32),
                {"overall": [1.0 for _ in range(len(data))]},
            )
        )
    )

    # build a tiny batch_dict to pass through _make_batch_data
    batch_dict = {
        "python_prompt": np.array(["s"], dtype=object),
        "source_prompt": np.array(["s"], dtype=object),
        "python_test": np.array(["gt"], dtype=object),
        "java_prompt": np.array(["sig"], dtype=object),
        "java_test": np.array(["jgt"], dtype=object),
    }
    trainer.train_dataloader = [batch_dict]
    trainer.data_iterator = iter(trainer.train_dataloader)
    trainer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    metrics = {}
    trainer._make_batch_data(metrics)

    assert len(exploration_pool) == 1
    assert exploration_pool[0]["source_prompt"] == "s"
    assert exploration_pool[0]["python_test"] == "gt"


if __name__ == "__main__":
    test_maybe_add_to_exploration()
