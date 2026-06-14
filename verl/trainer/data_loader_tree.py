# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataloader that supports a static seed pool and a dynamic exploration pool.
Exploration samples are always consumed first; remaining slots are filled
with seed samples, keeping the same behavior as the original dataloader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn, process_image, process_video, TreeRLHFDataset
from ..utils import torch_functional as VF
from .config import DataConfig


class _TreeBatchIterator:
    """Iterator that prioritizes exploration_pool then falls back to seed_pool."""

    def __init__(
        self,
        seed_dataset: RLHFDataset,
        sampler: Iterable[int],
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        exploration_pool: list[dict],
    ):
        self.seed_dataset = seed_dataset
        self.sampler_factory = sampler
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.exploration_pool = exploration_pool

        self._sampler_iter = iter(self.sampler_factory)

    def _reset_seed_iter(self):
        self._sampler_iter = iter(self.sampler_factory)

    def _process_example(self, example: dict) -> dict:
        # Adapted from RLHFDataset.__getitem__ to process an in-memory example.
        example = dict(example)
        messages = self.seed_dataset._build_messages(example)
        # example.pop(self.seed_dataset.prompt_key, None)

        if self.seed_dataset.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.seed_dataset.image_key)
            if (
                self.seed_dataset.image_dir is not None
                and len(images) != 0
                and isinstance(images[0], str)
            ):
                images = [Path(self.seed_dataset.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None
            for image in images:
                processed_images.append(process_image(image, self.seed_dataset.min_pixels, self.seed_dataset.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.seed_dataset.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.seed_dataset.video_key)
            if (
                self.seed_dataset.image_dir is not None
                and len(videos) != 0
                and isinstance(videos[0], str)
            ):
                videos = [Path(self.seed_dataset.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.seed_dataset.min_pixels,
                    self.seed_dataset.max_pixels,
                    self.seed_dataset.video_fps,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.seed_dataset.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.seed_dataset.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.seed_dataset.max_prompt_length:
            if self.seed_dataset.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.seed_dataset.max_prompt_length :]
            elif self.seed_dataset.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.seed_dataset.max_prompt_length]
            else:
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.seed_dataset.max_prompt_length}."
                )

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        # example["ground_truth"] = example.pop(self.seed_dataset.answer_key)
        return example

    def __iter__(self):
        return self

    def __next__(self):
        batch_items = []
        # take from exploration_pool first
        while len(batch_items) < self.batch_size and self.exploration_pool:
            batch_items.append(self.exploration_pool.pop(0))

        # fill remaining with seed pool
        while len(batch_items) < self.batch_size:
            try:
                idx = next(self._sampler_iter)
            except StopIteration:
                self._reset_seed_iter()
                idx = next(self._sampler_iter)
            batch_items.append(self.seed_dataset[idx])

        return collate_fn(batch_items)


class TreeDataLoader:
    """Dataloader with a static seed pool and a dynamic exploration pool."""

    def __init__(
        self,
        seed_dataset: RLHFDataset,
        batch_size: int,
        sampler,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        self.seed_dataset = seed_dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.processor = processor
        self.exploration_pool: list[dict] = []

    def add_exploration_data(self, examples: list[dict]):
        """Add raw examples into exploration pool (processed immediately)."""
        iterator = _TreeBatchIterator(
            self.seed_dataset,
            self.sampler,
            self.batch_size,
            self.tokenizer,
            self.processor,
            self.exploration_pool,
        )
        for example in examples:
            self.exploration_pool.append(iterator._process_example(example))

    def add_exploration_jsonl(self, path: str):
        """Load json/jsonl file and add to exploration pool."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Exploration file {path} not found.")

        examples = []
        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path_obj.suffix == ".jsonl":
                    examples.append(json.loads(line))
                else:
                    # json array
                    examples = json.load(f)
                    break
        self.add_exploration_data(examples)

    def __iter__(self):
        return _TreeBatchIterator(
            seed_dataset=self.seed_dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            processor=self.processor,
            exploration_pool=self.exploration_pool,
        )

    def __len__(self):
        return len(self.seed_dataset) // self.batch_size

    def state_dict(self) -> dict:
        """Minimal state for checkpointing."""
        sampler_state = None
        if hasattr(self.sampler, "generator") and self.sampler.generator is not None:
            sampler_state = self.sampler.generator.get_state()

        return {
            "exploration_pool": self.exploration_pool,
            "sampler_state": sampler_state,
        }

    def load_state_dict(self, state_dict: dict):
        self.exploration_pool = state_dict.get("exploration_pool", [])
        sampler_state = state_dict.get("sampler_state")

        if sampler_state is not None and hasattr(self.sampler, "generator") and self.sampler.generator is not None:
            self.sampler.generator.set_state(sampler_state)


def create_tree_dataloader(
    config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]
):
    """Create train/val dataloaders with tree-style pooling."""
    seed_dataset = TreeRLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )

    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=seed_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=seed_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = TreeDataLoader(
        seed_dataset=seed_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        tokenizer=tokenizer,
        processor=processor,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of seed train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
