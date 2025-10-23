# Taken and modified from https://github.com/huggingface/trl
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import logging
from openrlhf.utils.ifeval.ground_truth_utils import (
    verify_ifeval_sample,
)

logger = logging.getLogger(__name__)

def find_matching_prompt_tulu3(current_prompt, tulu3_ifeval_data_dict):

    if current_prompt in tulu3_ifeval_data_dict:
        ground_truth = tulu3_ifeval_data_dict[current_prompt]["ground_truth"]
        return True, ground_truth
        
    print(f"DEBUG_prompt_not_found:", current_prompt)
    return False, None  # Return None if no matching prompt is found

def apply_verifiable_reward(
    decoded_response: str,
    ground_truth: str,
    dataset: str,
):
    # compare with ground truth.
    verified = False
    if ground_truth is None:
        logger.warning("No ground truth provided for a sample, applying 0 reward.")
        return 0
    if dataset.lower() == "ifeval":
        verified = verify_ifeval_sample(decoded_response, ground_truth)
    # if verified, give reward
    if verified:
        logger.info("Applying ground truth reward 🤗")
        return 1.0
    
    return 0