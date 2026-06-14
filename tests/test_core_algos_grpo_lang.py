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

import torch

from verl.trainer.core_algos import compute_grpo_outcome_advantage_language


def test_compute_grpo_outcome_advantage_language_shapes():
    # two parents (p0, p1), each with two child groups (idx), rollout.n=2 per group
    # batch indices: [g0s0, g0s1, g1s0, g1s1]
    token_level_rewards = torch.tensor(
        [
            [1.0, 0.0],  # parent 0, group A, sample 1
            [2.0, 0.0],  # parent 0, group A, sample 2
            [4.0, 0.0],  # parent 0, group B, sample 1
            [6.0, 0.0],  # parent 0, group B, sample 2
        ]
    )
    response_mask = torch.tensor([[1, 0]] * 4, dtype=torch.float32)
    index = [0, 0, 1, 1] # group ids
    parent_uid = [10, 10, 10, 10]

    advantages, intra_adv = compute_grpo_outcome_advantage_language(
        token_level_rewards, response_mask, index, parent_uid=parent_uid
    )

    # advantages shape aligns with mask
    assert advantages.shape == token_level_rewards.shape
    # intra_adv is (bs,)
    assert intra_adv.shape == (4,)

    # weights: parent sum = (1+2)+(4+6)=13, group0 sum=3, group1 sum=10
    # for group0 samples: A=3, B=10 -> w=10/13; group1: A=10, B=3 -> w=3/13
    expected = torch.tensor([10 / 13, 10 / 13, 3 / 13, 3 / 13], dtype=torch.float32)
    assert torch.allclose(intra_adv, expected, atol=1e-5)


def test_compute_grpo_outcome_advantage_language_parent_missing():
    token_level_rewards = torch.ones((2, 1))
    response_mask = torch.ones_like(token_level_rewards)
    index = torch.tensor([0, 0])
    try:
        compute_grpo_outcome_advantage_language(token_level_rewards, response_mask, index)
    except RuntimeError as e:
        assert "parent_uid" in str(e)
    else:
        assert False, "expected RuntimeError when parent_uid is missing"


if __name__ == "__main__":
    test_compute_grpo_outcome_advantage_language_shapes()
    test_compute_grpo_outcome_advantage_language_parent_missing()
