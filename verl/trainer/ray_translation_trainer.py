
"""
Ray PPO trainer for multi-language code translation.
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import ray
import torch
from jinja2 import Template

from ..protocol import DataProto
from ..utils import torch_functional as VF
from .metrics import reduce_metrics
from .ray_trainer import RayPPOTrainer


def markdown_codeblock_extract(new: str) -> str:
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


def post_process(new: str) -> str:
    try:
        extracted = markdown_codeblock_extract(new)
    except Exception as e:
        print(f"Failed to extract codeblock from {new}: {e}")
        extracted = new
    return extracted.strip()

class RayCodeTranslationPPOTrainer(RayPPOTrainer):
    _DEFAULT_SOURCE_LANG = "python"

    def _get_translation_prompt_template(self) -> Template:
        if getattr(self, "_translation_prompt_template", None) is None:
            if not self.config.data.format_prompt:
                raise ValueError("config.data.format_prompt must point to a translation prompt template.")
            with open(self.config.data.format_prompt, encoding="utf-8") as f:
                self._translation_prompt_template = Template(f.read().strip())
        return self._translation_prompt_template

    def _extract_prompt_langs(self, batch_dict: dict[str, Any]) -> list[str]:
        prompt_suffix = "_prompt"
        langs = {key[: -len(prompt_suffix)] for key in batch_dict if key.endswith(prompt_suffix)}
        return sorted(langs)

    def _tokenize_prompts(self, prompt_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        if len(prompt_texts) == 0:
            raise RuntimeError("Empty prompt list for translation batch.")

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []
        max_prompt_length = self.config.data.max_prompt_length

        for prompt_text in prompt_texts:
            messages = [{"role": "user", "content": prompt_text}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )

            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            raw_prompt_ids_list.append(raw_prompt_ids)

        raw_prompt_ids_array = np.empty(len(raw_prompt_ids_list), dtype=object)
        for i, ids in enumerate(raw_prompt_ids_list):
            raw_prompt_ids_array[i] = np.array(ids, dtype=np.int64)

        return (
            torch.stack(input_ids_list, dim=0),
            torch.stack(attention_mask_list, dim=0),
            torch.stack(position_ids_list, dim=0),
            raw_prompt_ids_array,
        )

    def _build_translation_batch(self, batch_dict: dict[str, Any], meta_info: dict[str, Any]) -> DataProto:
        source_prompts = batch_dict["source_prompt"]
        batch_size = len(source_prompts)
        parent_uids = np.array(
                        [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
                    )
        if "source_language" in batch_dict:
            src_raw = np.array(batch_dict["source_language"], dtype=object)
            source_langs = np.array([self._DEFAULT_SOURCE_LANG] * batch_size, dtype=object)
            source_langs[: min(len(src_raw), batch_size)] = src_raw
        else:
            source_langs = np.array([self._DEFAULT_SOURCE_LANG] * batch_size, dtype=object)
        target_langs_all = self.config.translation.prompt_langs

        if "depth" in batch_dict:
            depth_raw = batch_dict["depth"]
            depth_array = np.zeros(batch_size, dtype=object)
            depth_array[: min(len(depth_raw), batch_size)] = np.array(depth_raw, dtype=object)
            depth_array = depth_array + 1
        else:
            depth_array = np.ones(batch_size, dtype=object)

        template = self._get_translation_prompt_template()
        all_samples = []
        for idx in range(batch_size):
            src_lang = source_langs[idx]
            tgt_langs = [lang for lang in target_langs_all if lang != src_lang]
            if not tgt_langs:
                continue
            for tgt_lang in tgt_langs:
                if f"{tgt_lang}_prompt" not in batch_dict:
                    continue
                tgt_signature = batch_dict[f"{tgt_lang}_prompt"][idx]
                prompt_text = template.render(
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    source_prompt=source_prompts[idx],
                    target_signature=tgt_signature,
                )
                all_samples.append((idx, src_lang, tgt_lang, prompt_text))

        if not all_samples:
            raise RuntimeError("No translation samples constructed from batch_dict.")

        prompt_texts = [s[3] for s in all_samples]
        input_ids, attention_mask, position_ids, raw_prompt_ids = self._tokenize_prompts(prompt_texts)

        expanded_indices = [s[0] for s in all_samples]

        def _is_per_sample_field(value: Any) -> bool:
            if isinstance(value, (str, bytes)):
                return False
            try:
                return len(value) == batch_size
            except TypeError:
                return False

        def _gather_per_sample(value: Any) -> np.ndarray:
            gathered = [value[idx] for idx in expanded_indices]
            arr = np.empty(len(gathered), dtype=object)
            for i, item in enumerate(gathered):
                arr[i] = item
            return arr

        non_tensors = {}
        for key, value in batch_dict.items():
            if key in {"depth", "raw_prompt_ids", "source_language", "target_language"}:
                continue
            # if key.endswith("_prompt") and key != "source_prompt":
            #     continue
            if not _is_per_sample_field(value):
                continue
            non_tensors[key] = _gather_per_sample(value)

        non_tensors["source_language"] = np.array([s[1] for s in all_samples], dtype=object)
        non_tensors["target_language"] = np.array([s[2] for s in all_samples], dtype=object)
        non_tensors["depth"] = np.array([depth_array[idx] for idx in expanded_indices], dtype=object)
        # # attach ground truth/test per target lang
        # gt_list = []
        # tgt_tests = {}
        # for idx, _, tgt_lang, _ in all_samples:
        #     test_key = f"{tgt_lang}_test"
        #     if test_key not in batch_dict:
        #         raise RuntimeError(f"Missing `{test_key}` in batch for translation reward.")
        #     gt_list.append(batch_dict[test_key][idx])
        #     if test_key not in tgt_tests:
        #         tgt_tests[test_key] = []
        #     tgt_tests[test_key].append(batch_dict[test_key][idx])
        # non_tensors["ground_truth"] = np.array(gt_list, dtype=object)
        # for key, val in tgt_tests.items():
        #     non_tensors[key] = np.array(val, dtype=object)
        non_tensors["raw_prompt_ids"] = raw_prompt_ids
        non_tensors["parent_uids"] =  np.array([parent_uids[idx] for idx in expanded_indices], dtype=object)

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _maybe_add_to_exploration(self, batch: DataProto, indices: list[int]) -> None:
        """Add selected samples into exploration pool if supported by dataloader."""
        if not hasattr(self.train_dataloader, "add_exploration_data"):
            return

        response_ids = batch.batch["responses"]
        response_length = torch.sum(batch.batch["response_mask"], dim=-1)
        responses = []
        for i in range(len(batch)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            # TODO 从response_str中抽取代码
            code = post_process(response_str)
            responses.append(code)
        
        examples = []
        for idx in indices:
            item = batch[idx]
            # TODO 判断数据的depth是否大于self.config.translation.tree_depth if item.non_tensor_batch['depth'] >= self.config.translation.tree_depth:\n continue
            if item.non_tensor_batch['depth'] >= self.config.translation.tree_depth:
                continue
            record = {}
            for key, value in item.non_tensor_batch.items():
                # keep lightweight metadata; drop large token lists
                if key in {"raw_prompt_ids", "multi_modal_data", 'input_ids', 'attention_mask', 'position_ids', 'uid', 'parent_uids', 'source_language'}:
                    continue
                elif key == 'source_prompt':
                    record[key] = responses[idx]
                elif key == 'target_language':
                    record['source_language'] = value
                elif isinstance(value, np.ndarray):
                    if value.shape == ():
                        record[key] = value.item()
                    else:
                        record[key] = value.tolist()
                else:
                    record[key] = value

            examples.append(record)

        if examples:
            self.train_dataloader.add_exploration_data(examples)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = self._build_translation_batch(batch_dict, meta_info=meta_info)

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)
                filter_scores = reward_tensor.sum(axis=1)
                # filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                uid2indices = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)
                for idx, uid in enumerate(uids):
                    uid2indices[uid].append(idx)

                # if any sample of the same uid has reward 1, randomly add one to exploration pool
                exploration_indices = []
                for uid, scores in uid2scores.items():
                    if any(s == 1 for s in scores):
                        exploration_indices.append(np.random.choice(uid2indices[uid]))
                if exploration_indices:
                    self._maybe_add_to_exploration(new_batch, exploration_indices)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    continue
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size * self.config.translation.tree_width:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n * self.config.translation.tree_width]
