

import types

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.trainer.config import DataConfig
from verl.trainer.data_loader import create_dataloader
from verl.trainer.ray_translation_trainer import RayCodeTranslationPPOTrainer


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


class DummyRolloutWorkerGroup:
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = len(prompts)
        resp_len = 3
        responses = torch.full((batch_size, resp_len), 7, dtype=torch.long)
        response_mask = torch.ones_like(responses)
        batch = {
            "input_ids": prompts.batch["input_ids"],
            "attention_mask": prompts.batch["attention_mask"],
            "position_ids": prompts.batch["position_ids"],
            "responses": responses,
            "response_mask": response_mask,
        }
        return DataProto.from_dict(tensors=batch, non_tensors={}, meta_info=prompts.meta_info)


def _build_config(template_path: str, rollout_batch_size: int):
    data = DataConfig(
        train_files="",
        val_files="",
        prompt_key="source_prompt",
        answer_key="python_test",
        max_prompt_length=1024,
        rollout_batch_size=rollout_batch_size,
        mini_rollout_batch_size=2,
        format_prompt=None,
        shuffle=False,
        filter_overlong_prompts=False,
        val_batch_size=2,
    )
    worker = types.SimpleNamespace(rollout=types.SimpleNamespace(n=1))
    algorithm = types.SimpleNamespace(adv_estimator="grpo", online_filtering=False)
    trainer = types.SimpleNamespace(max_try_make_batch=1)
    return types.SimpleNamespace(data=data, worker=worker, algorithm=algorithm, trainer=trainer)


def test_make_batch_data_translation(tmp_path, train_file, tokenizer_path):
    template_path = Path(tmp_path) / "translation.jinja"
    templlate_str = """Please translate source {{source_lang}} code to {{target_lang}} code:\n```{{source_lang}}\n{{source_prompt}}```\nThe translated {{target_lang}} code should be:\n```{{target_lang}}\n{{target_signature}}```"""

    template_path.write_text(templlate_str)

    data_cfg = _build_config(str(template_path), rollout_batch_size=4).data
    data_cfg.train_files = str(train_file)
    data_cfg.val_files = str(train_file)
    
    tokenizer = DummyTokenizer()
    train_dataloader, _ = create_dataloader(data_cfg, tokenizer, processor=None)
    
    trainer = RayCodeTranslationPPOTrainer.__new__(RayCodeTranslationPPOTrainer)
    # trainer.tokenizer = DummyTokenizer()
    trainer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    trainer.config = _build_config(str(template_path), rollout_batch_size=4)
    trainer.config.data.format_prompt = str(template_path)
    trainer.reward_fn = None
    trainer.actor_rollout_ref_wg = DummyRolloutWorkerGroup()
    
    trainer.train_dataloader = train_dataloader
    trainer.data_iterator = iter(trainer.train_dataloader)

    metrics = {}
    batch = trainer._make_batch_data(metrics=metrics)
    
    # assert len(batch) == 4
    # assert batch.non_tensor_batch["source_lang"].tolist() == ["python"] * 4
    # assert batch.non_tensor_batch["tgt_lang"].tolist() == ["go", "go", "java", "java"]
    # assert batch.non_tensor_batch["ground_truth"].tolist() == ["go_gt1", "go_gt2", "java_gt1", "java_gt2"]
    # assert torch.all(batch.batch["responses"] == 7)
    # assert "raw_prompt_ids" in batch.non_tensor_batch


if __name__ == "__main__":
    test_make_batch_data_translation(tmp_path="examples/format_prompt", train_file="examples/train_data.jsonl", tokenizer_path="/data1/shares/Qwen2.5-7B-Instruct")
