import functools
import itertools
import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
import tree
from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import DataLoader

from datasets import load_from_disk
from understand_r1_zero.math_grader import (answer_tag_reward_fn,
                                            boxed_reward_fn)

import time
import logging
import itertools
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import tree
import math

"""
1. To do RL from base models, we use proper prompt template to make the base model answer questions.
"""


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_no_template(question: str):
    return question


TEMPLATE_FACTORY = {
    "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
}


"""
2. To train reasoning models that solve math questions, we need to define an oracle (environment) that provides rule-based verification rewards.
We instantiate the oracle based on Oat's OracleBase and implement the grading logic.
"""


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(self, template, verifier_version) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        else:
            math_reward_fn = boxed_reward_fn
        self.math_reward_fn = functools.partial(
            math_reward_fn, fast=verifier_version == "fast"
        )
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        # Parameters used by Oat when using model-based reward, here we don't need.
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, r = res.get(timeout=1)
                rewards.append(r)
                infos.append(info)
            except TimeoutError:
                rewards.append(0.0)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


"""
2. Define extra arguments needed besides Oat's PPOArgs, mainly about choosing the prompt template.
"""


@dataclass
class ZeroMathArgs(PPOArgs):
    # Template.
    prompt_template: Literal["qwen_math", "no", "r1"] = field(default="qwen_math")
    # Evaluation benchmarks used.
    test_split: str = "all"  # Use "aime,math" to only evaluate on selected benchmarks.
    # Verifier.
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")

    drgrpo_p: float = field(
        default=0.00, metadata={"help": "The 'p' parameter for Dr. GRPO weighting."}
    )
    max_tries:int = field(
        default=0, metadata={"help": "number of retrying sample(include the initial one)."}
    )


"""
3. Instantiate the actor based on Oat's PPOActor, which controls the reasoning trace generation (`self.sampling_params`) and the rewarding (`self.oracle`).
"""

_group_weight_cache = {}

def group_weight(n: int, k: int, p: float, epsilon: float = 1e-8) -> float:
    """
    Calculate the optimal weight w* of S-GRPO.

    This function implements the following formula:
    w*(n, k, p) = ((1 - 2p) * t * (1 - t)) / (sqrt(r_bar * (1 - r_bar) + eps) * sqrt(t * (1 - t) + eps))

    Args:
        n (int): The total number of samples in the group (group size).
        k (int): Number of observed successes (reward = 1) in the group.
        p (float): Symmetric reward flipping probability (noise probability), must be in [0, 0.5).
        epsilon (float): A small constant for numerical stability.

    Returns:
        float: The optimal weight w*.
    """
    # --- 1. Input validation and edge case handling ---
    if not (0 <= p < 0.5):
        raise ValueError("Noise probability p must be in the range [0, 0.5).")

    if n == 0:
        return 0.0  # If there are no samples, the weight is not meaningful.

    if not (0 <= k <= n):
        raise ValueError("The number of successes k must be between 0 and n.")

    # --- 2. Compute intermediate variables ---
    # Compute the observed average reward r_bar (i.e., q)
    r_bar = k / n

    # Estimate the true average reward t, clipped to the range [0, 1]
    # Since p < 0.5, the denominator (1 - 2*p) is guaranteed to be positive
    if (1 - 2 * p) == 0:  # Avoid division by zero
        t_raw = 0
    else:
        t_raw = (r_bar - p) / (1 - 2 * p)

    t = max(0.0, min(1.0, t_raw))

    # --- 3. Compute the final weight w* ---
    # Compute the numerator
    numerator = (1 - 2 * p) * t * (1 - t)

    # Compute the two components of the denominator
    # Standard deviation of observed reward
    std_dev_r = math.sqrt(r_bar * (1 - r_bar) + epsilon)
    # Standard deviation of true reward
    std_dev_t = math.sqrt(t * (1 - t) + epsilon)

    # Combine denominator
    denominator = std_dev_r * std_dev_t

    if denominator == 0:
        return 0.0  # Avoid division by zero

    # Compute the final weight
    weight = numerator / denominator

    return weight

    
import random
class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroMathArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)

        self.oracle = MATHOracle(
            template=args.prompt_template, verifier_version=args.verifier_version
        )

        if args.prompt_template in ["qwen_math", "no"]:
            # These two templates are better used for Qwen models, which can themselves stop generation. Hence we unset all external stopping conditions.
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif args.prompt_template == "r1":
            # Let's stop when the model completes its answer.
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

    def _generate_and_unpack(self, formatted_prompts: List[str]) -> Dict[str, Any]:
        """
        Generates text for a list of prompts and unpacks the output into a structured dictionary.
    
        Args:
            formatted_prompts: A list of prompts to generate from.
    
        Returns:
            A dictionary where each key is a formatted prompt and the value is a dictionary
            containing the prompt's token IDs and a list of its generated samples.
        """
        outputs = self.generate(formatted_prompts, self.sampling_params)
    
        results_dict = {}
        for i, prompt_text in enumerate(formatted_prompts):
            # Data for this specific prompt
            prompt_output_data = {
                "prompt_token_ids": outputs[i].prompt_token_ids,
                "samples": []
            }
    
            # Iterate through the 'n' samples generated for this prompt
            for k in range(self.sampling_params.n):
                generation_output = outputs[i].outputs[k]
                token_ids = generation_output.token_ids
    
                # Unpack log probabilities for the generated tokens
                raw_logprobs = generation_output.logprobs
                logprobs = [item[token_ids[idx]].logprob for idx, item in enumerate(raw_logprobs)]
    
                # Store all useful information for this single sample
                sample_data = {
                    "candidate": generation_output.text,
                    "no_eos": generation_output.finish_reason == "length",
                    "response_ids": token_ids,
                    "response_logprobs": logprobs,
                    "response_len": len(token_ids)
                }
                prompt_output_data["samples"].append(sample_data)
            
            results_dict[prompt_text] = prompt_output_data
    
        return results_dict

    def _get_and_assign_rewards(
        self,
        results_dict: Dict[str, Any],
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates rewards for all generated candidates and updates the results dictionary.
    
        Args:
            results_dict: The dictionary from _generate_and_unpack, containing all generated samples.
            prompts: The original, unformatted prompts.
            formatted_prompts: The formatted prompts, used as keys for the results_dict.
            references: A list of reference texts, parallel to the prompts.
    
        Returns:
            The results_dict, updated in-place with a 'reward' and 'oracle_info'
            for each generated sample.
        """
        n = self.sampling_params.n
    
        # 1. Flatten all candidate strings from the results_dict in the correct order.
        # The order must match the order of the formatted_prompts list.
        flat_candidates = [
            sample["candidate"]
            for prompt_text in formatted_prompts
            for sample in results_dict[prompt_text]["samples"]
        ]
    
        # 2. Repeat prompts and references to match the flattened candidate list.
        repeated_prompts = list(
            itertools.chain.from_iterable(itertools.repeat(x, n) for x in prompts)
        )
    
        repeated_references = None
        if references:
            repeated_references = list(
                itertools.chain.from_iterable(itertools.repeat(x, n) for x in references)
            )
    
        # 3. Call the oracle to get rewards for all candidates at once.
        rewards, oracle_infos = self.oracle.get_reward(
            repeated_prompts,
            flat_candidates,
            repeated_references,
        )
    
        # 4. Reshape rewards to match the original structure (num_prompts, n_samples).
        rewards = rewards.reshape(len(prompts), n)
    
        # 5. Iterate through the results_dict and assign the corresponding reward
        # and oracle_info to each sample.
        for i, prompt_text in enumerate(formatted_prompts):
            for j in range(n):
                # Calculate the flat index to access the flat oracle_infos list
                flat_index = i * n + j
                
                # Update the specific sample in the dictionary
                sample_to_update = results_dict[prompt_text]["samples"][j]
                sample_to_update["reward"] = rewards[i][j].item()
                sample_to_update["oracle_info"] = oracle_infos[flat_index]
    
        return results_dict
        
    def _create_trajectory_data(
        self,
        results_dict: Dict[str, Any],
        prompts: List[str],
        formatted_prompts: List[str],
        info: Dict[str, Any],
    ) -> List[Any]: # The return type is List[TrajectoryData]
        """
        Constructs the final list of TrajectoryData objects from the results dictionary.
    
        Args:
            results_dict: The dictionary containing all sample data, including rewards.
            prompts: The original, unformatted prompts.
            formatted_prompts: The formatted prompts, used as keys for the results_dict.
            info: A dictionary for logging that will be attached to each trajectory.
    
        Returns:
            A list of TrajectoryData objects.
        """
        if len(results_dict) == 0:
            return []
        trajectory_data = []
    
        # Iterate through prompts in their original order to maintain consistency
        for i, prompt_text in enumerate(prompts):
            formatted_key = formatted_prompts[i]
            if formatted_key not in results_dict:
                continue
            prompt_output_data = results_dict[formatted_key]
            
            prompt_ids = prompt_output_data["prompt_token_ids"]
    
            # Iterate through each sample generated for the current prompt
            for sample in prompt_output_data["samples"]:
                reward = sample["reward"]
                is_no_eos = sample["no_eos"]
                response_ids = sample["response_ids"]
    
                # Per the original logic, set the reward to 0 for truncated outputs
                if is_no_eos:
                    reward = 0
    
                # Create the dense_rewards list, where the reward is only at the last step
                dense_rewards = [0] * len(response_ids)
                if dense_rewards:
                    dense_rewards[-1] = reward
    
                # --- FIX ---
                # The error occurred here. `self.args` is a class instance, not a dict.
                # Replace `self.args.get(...)` with a safe check using `hasattr()`.
                should_ignore_no_eos = (
                    hasattr(self.args, "ignore_no_eos") and self.args.ignore_no_eos
                )
                loss_mask = not is_no_eos if should_ignore_no_eos else True
                # --- END FIX ---
    
                # Append the fully formed TrajectoryData object
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt_text,
                        prompt_ids=prompt_ids,
                        response=sample["candidate"],
                        response_ids=response_ids,
                        response_logprobs=sample["response_logprobs"],
                        rewards=dense_rewards,
                        loss_mask=loss_mask,
                        info=info,
                    )
                )
                
        return trajectory_data
    def _select_most_balanced_n_samples(self, all_samples: List[Dict]) -> List[Dict]:
        """
        From a pool of samples, selects n samples to be as balanced as possible.
        This version uses a clear, case-based logic for improved readability.
    
        Args:
            all_samples: A list of all available sample dictionaries for a prompt.
                         It is assumed that len(all_samples) >= n.
    
        Returns:
            A new list containing exactly n samples, selected for balance.
        """
        n = self.sampling_params.n
    
        # 1. Separate samples and get counts.
        pos_samples = sorted(
            [s for s in all_samples if s['reward'] > 0],
            key=lambda x: x['reward'],
            reverse=True
        )
        zero_samples = [s for s in all_samples if s['reward'] == 0]
        
        num_pos = len(pos_samples)
        num_zero = len(zero_samples)
        
        target_pos_count = n // 2
        target_zero_count = n - target_pos_count # Handles odd 'n' correctly
    
        selected_samples = []
    
        # 2. Determine which case we are in.
        if num_pos >= target_pos_count and num_zero >= target_zero_count:
            # Case 1: Ideal Balance. We have enough of both.
            # Take the best positive samples and a selection of zero-reward samples.
            selected_samples.extend(pos_samples[:target_pos_count])
            selected_samples.extend(zero_samples[:target_zero_count])
            
        elif num_pos < target_pos_count:
            # Case 2: Not enough positive samples.
            # Take all available positive samples and fill the rest with zero-reward ones.
            selected_samples.extend(pos_samples)
            remaining_needed = n - num_pos
            selected_samples.extend(zero_samples[:remaining_needed])
            
        else: # This implies num_zero < target_zero_count
            # Case 3: Not enough zero-reward samples.
            # Take all available zero-reward samples and fill the rest with the best positive ones.
            selected_samples.extend(zero_samples)
            remaining_needed = n - num_zero
            selected_samples.extend(pos_samples[:remaining_needed])
            
        # The list is guaranteed to have exactly n samples due to the logic above
        # and the assumption that len(all_samples) >= n.
        return selected_samples


    def _check_finished(self, results_to_check: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Partitions a results dictionary into finished and unfinished sets based on a new logic.
    
        For each prompt:
        1. Selects the "most balanced" subset of n samples from all available samples.
        2. Checks if this ideal subset is balanced (has a mix of rewards).
        3. If balanced, the prompt is moved to `finished_dict` with ONLY the n selected samples.
        4. If not, the prompt is moved to `unfinished_dict` with ALL of its original samples.
        """
        finished_dict = {}
        unfinished_dict = {}
        n = self.sampling_params.n
    
        for prompt_text, data in results_to_check.items():
            all_samples_for_prompt = data["samples"]
    
            # 1. Pickup the most balanced top n samples using the helper function.
            best_n_samples = self._select_most_balanced_n_samples(all_samples_for_prompt)
    
            # 2. Check if this ideal subset passes the balance criteria.
            pos_count = sum(1 for s in best_n_samples if s['reward'] > 0)
            is_balanced = (group_weight(n,pos_count,self.args.drgrpo_p) != 0)
    
            if is_balanced:
                # 3. This prompt is finished. Retain ONLY the best n samples.
                finished_data = data.copy() # Creates a shallow copy of the prompt-level data
                finished_data['samples'] = best_n_samples
                finished_dict[prompt_text] = finished_data
            else:
                # 4. This prompt is still unfinished. Keep ALL its samples for the next round.
                unfinished_dict[prompt_text] = data
                
        return finished_dict, unfinished_dict

    def _combine_results(
        self, old_results: Dict[str, Any], new_results: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Combines newly generated samples with the existing samples for each prompt.
    
        Args:
            old_results: The dictionary of unfinished prompts, potentially with many samples.
            new_results: The dictionary containing the newly generated samples from resampling.
    
        Returns:
            A new dictionary where each prompt's 'samples' list now contains both the
            old and the new samples.
        """
        # It's good practice to work on a copy to avoid modifying the original dict in-place.
        combined_results = old_results.copy()
    
        # Iterate through the prompts that have newly generated results
        for prompt_text, new_data in new_results.items():
            
            # Ensure the prompt from the new results exists in the old results before combining
            if prompt_text in combined_results:
                newly_generated_samples = new_data["samples"]
                
                # Append the new samples to the existing list of samples for that prompt
                combined_results[prompt_text]["samples"].extend(newly_generated_samples)
            else:
                # This case should ideally not happen if logic is correct, but it's safe to handle
                # It means a new prompt appeared that wasn't in the original "unfinished" set
                combined_results[prompt_text] = new_data
    
        return combined_results
        
    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> Any: # The return type depends on the IPC client
        """
        Main logic for the actor to generate trajectories using a modular approach.
        """
        assert not self.eval_mode
        info = {}
        logging.info(f"Actor starting step...")
    
        # Step 1. Generate candidates and unpack the results into a structured dictionary.
        st = time.time()
        results_dict = self._generate_and_unpack(formatted_prompts)
        info["actor/generate_time"] = time.time() - st
    
        # Step 2. Get rewards for the generated candidates and update the dictionary.
        st = time.time() # Reset timer for the verification step
        results_dict = self._get_and_assign_rewards(
            results_dict, prompts, formatted_prompts, references
        )
        info["actor/verify_time"] = time.time() - st

        # --- Start of Resampling Block ---
        st_resample = time.time()
        logging.info("Checking for imbalanced results to start resampling...")
        
        # To correctly call the reward function inside the loop, we need a map
        # from the formatted prompt (dict key) back to the original prompt and reference.
        prompt_map = {
            f: {"prompt": p, "ref": r}
            for f, p, r in zip(formatted_prompts, prompts, references or [None] * len(prompts))
        }
        
        finished_results_dict, unfinished_results_dict = self._check_finished(results_dict)
        initial_finished_count = len(finished_results_dict)
        logging.info(
            f"Initial check: {len(finished_results_dict)} finished, "
            f"{len(unfinished_results_dict)} require resampling."
        )
        
        max_tries = self.args.max_tries # The initial attempt + (max_tries - 1) retries
        print('max_tries',max_tries)
        for i_try in range(max_tries - 1):
            # If all prompts are finished, no need to continue.
            if not unfinished_results_dict:
                logging.info("All prompts are now balanced. Stopping resampling.")
                break
        
            logging.info(f"Resampling round {i_try + 1} for {len(unfinished_results_dict)} prompts.")
            
            # 1. Prepare inputs for the unfinished prompts
            unfinished_formatted_prompts = list(unfinished_results_dict.keys())
            unfinished_prompts = [prompt_map[f]["prompt"] for f in unfinished_formatted_prompts]
            unfinished_refs = [prompt_map[f]["ref"] for f in unfinished_formatted_prompts]
            if all(r is None for r in unfinished_refs):
                unfinished_refs = None
        
            # 2. Generate new samples for ONLY the unfinished prompts
            new_results_dict = self._generate_and_unpack(unfinished_formatted_prompts)
        
            # 3. Get rewards for ONLY the new samples
            new_results_dict = self._get_and_assign_rewards(
                new_results_dict, unfinished_prompts, unfinished_formatted_prompts, unfinished_refs
            )
        
            # 4. Combine new results with the old ones using a replacement strategy
            unfinished_results_dict = self._combine_results(
                unfinished_results_dict, new_results_dict
            )
        
            # 5. Re-check the updated prompts and partition them again
            newly_finished, unfinished_results_dict = self._check_finished(
                unfinished_results_dict
            )
        
            # 6. Add the newly finished prompts to the main finished dictionary
            if newly_finished:
                logging.info(f"Round {i_try + 1}: {len(newly_finished)} more prompts are now finished.")
                finished_results_dict.update(newly_finished)
                
        for prompt_text, data in unfinished_results_dict.items():
            all_accumulated_samples = data["samples"]
            
            # Use the helper function to select the most balanced subset of size 'n'
            best_n_samples = self._select_most_balanced_n_samples(all_accumulated_samples)
            
            # Replace the inflated list with the curated list of 'n' samples
            unfinished_results_dict[prompt_text]["samples"] = best_n_samples
        
        # --- Now the merge is safe ---
        # Both dictionaries now contain prompts with exactly 'n' samples each.

        final_finished_count = len(finished_results_dict)
        logging.info(
            f"Resampling complete. Finished prompt count increased from "
            f"{initial_finished_count} to {final_finished_count}."
        )
        results_dict = finished_results_dict
        results_dict.update(unfinished_results_dict)  
        
        info["actor/resample_time"] = time.time() - st_resample
        # --- End of Resampling Block ---
    
        # Step 3. Calculate and log aggregate statistics from the final results_dict.
        # We must extract the data from the dictionary before calculating stats.
        all_rewards = []
        all_oracle_infos = []
        all_resp_lens = []
        all_no_eos_flags = []
        if len(results_dict)>0:
            for prompt_text in formatted_prompts:
                if prompt_text not in results_dict:
                    continue
                for sample in results_dict[prompt_text]["samples"]:
                    all_rewards.append(sample["reward"])
                    all_oracle_infos.append(sample["oracle_info"])
                    all_resp_lens.append(sample["response_len"])
                    all_no_eos_flags.append(sample["no_eos"])
    
        # Now, calculate statistics using the collected lists.
        if all_rewards:
            mean_reward = np.mean(all_rewards)
            logging.info(f"Actor mean reward: {mean_reward:.4f}")
            info["actor/rewards"] = mean_reward
            info["actor/num_data"] = len(all_rewards)
            info["actor/no_eos_count"] = np.sum(all_no_eos_flags)
            info["actor/response_tok_len"] = np.mean(all_resp_lens)
            
            # Safely calculate formatted ratio from oracle_infos
            if all_oracle_infos and isinstance(all_oracle_infos[0], dict):
                formatted_flags = [oi.get("formatted", 0.0) for oi in all_oracle_infos]
                info["actor/formatted"] = np.mean(formatted_flags)
    
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        import random
        # For each prompt, randomly shuffle the sample list within it once
        for prompt_key in results_dict:
            random.shuffle(results_dict[prompt_key]["samples"])

    
        # Step 4. Create the final TrajectoryData objects for training.
        trajectory_data = self._create_trajectory_data(
            results_dict,
            prompts,
            formatted_prompts,
            info,
        )
        
        logging.info(f"Actor finished. Final data length: {len(trajectory_data)}")
        
        # Step 5. Serialize and return the data.
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
     
 
class ZeroMathLearner(PPOLearner):
    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.eval_dataset_dict = load_from_disk(args.eval_data)  # TODO: get fro HF.
        if args.test_split != "all":
            self.eval_dataset_dict = {
                k: v for k, v in self.eval_dataset_dict.items() if k in args.test_split
            }
        self.args = args
        # Dr. GRPO Modification 1: Remove length bias by using masked_sum with a constant normalizer:
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        
    

    # Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std:
    def compute_monte_carlo_advantages(self, rewards):
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        
        grouped_rewards = rewards.view(-1, self.args.num_samples)
        
        advantages = rewards - values
        if self.args.critic_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        elif self.args.critic_type == "drgrpo":
            # Dr. GRPO Modification 2: Apply w* weighting.
            # Calculate k (number of successes) for each prompt. Assuming success is reward > 0.
            k_per_prompt = (grouped_rewards > 0).sum(dim=1).float()
            
            # Get N and p from arguments
            N = self.args.num_samples
            p = self.args.drgrpo_p
            # Calculate weight for each prompt
            weights = torch.zeros_like(k_per_prompt, device=rewards.device)
            no_zero_count = []
            for i in range(len(k_per_prompt)):
                k = k_per_prompt[i].item()
                no_zero_count.append(k)
                weights[i] = group_weight(N, int(k), p)

            # Reshape weights to match advantages shape and apply them
            print('weights',weights)
            
            weights_repeated = weights.repeat_interleave(self.args.num_samples, dim=0)
            
            
            print('no_zero_count',no_zero_count)
            print('p',p)

            #recover to grpo
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages =  advantages * weights_repeated / (std_grouped_rewards + 1e-8)
        return advantages

    def _apply_template(self, example):
        problem = example[self.args.input_key]
        example[self.args.input_key] = TEMPLATE_FACTORY[args.prompt_template](problem)
        return example

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )

        # Prepare the data: templated questions & gt final answers.
        prompts_data = prompts_data.map(lambda x: self._apply_template(x))

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def eval_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def evaluate(self, dataloader, steps):
        print('steps',steps)
        if steps == 0:
            print('0 steps return')
            return {
                "eval/average/accuracy": -1,
                "eval/average/score": -1,
                "eval/average/response_tok_len": -1,
            }
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )
        return all_metrics

import os
import copy

def remap_cuda_visible_devices(proc_dict):
    visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_list = [x.strip() for x in visible_str.split(",") if x.strip().isdigit()]
    id_map = {str(i): visible_list[i] for i in range(len(visible_list))}

    new_dict = {}

    for name, proc in proc_dict.items():
        # Create a new instance of PythonProcess (deeply copy all properties)
        new_proc = copy.deepcopy(proc)

        if hasattr(new_proc, "env") and "CUDA_VISIBLE_DEVICES" in new_proc.env:
            logical_id = new_proc.env["CUDA_VISIBLE_DEVICES"]
            if logical_id in id_map:
                new_proc.env["CUDA_VISIBLE_DEVICES"] = id_map[logical_id]

        new_dict[name] = new_proc

    return new_dict
    
def run_zero_math_rl(args: ZeroMathArgs):
    # Define a distributed program that composes Actors and Learners.
    program, local_resources = get_program(
        args, learner_cls=ZeroMathLearner, actor_cls=ZeroMathActor
    )
    print(local_resources)
    import os

    local_resources = remap_cuda_visible_devices(local_resources)
    print(local_resources)
    # Launch the program in a local, multi-processing way!
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: ZeroMathArgs = get_default_args(ZeroMathArgs)
    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    run_zero_math_rl(args)