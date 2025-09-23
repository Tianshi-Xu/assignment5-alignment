from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from unittest.mock import patch
import torch
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from typing import List, Dict, Iterable
from eval_math import load_math_dataset, format_prompt, evaluate_vllm, r1_zero_reward_fn


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    # with world_size_patch:
    return LLM(model=model_id,device=device,dtype=torch.bfloat16, enable_prefix_caching=True,gpu_memory_utilization=gpu_memory_utilization,max_num_seqs=256)


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
def get_loader(dataset: List[Dict], batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def grad_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    l2_norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        l2_norm += torch.sum(param.grad.data ** 2)
    l2_norm = torch.sqrt(l2_norm)
    for param in parameters:
        if param.grad is None:
            continue
        if l2_norm > max_l2_norm:
            param.grad.data = param.grad.data * max_l2_norm/(l2_norm + 1e-8)
    return l2_norm

@torch.no_grad()
def evaluate(args,logger,model:LLM) -> float:
    prompt_file = "cs336_alignment/prompts/r1_zero.prompt"
    jsonl_file = args.valid_dir
    examples = load_math_dataset(jsonl_file)
    prompts = [format_prompt(prompt_file, example['problem']) for example in examples[:1024]]
    # prompts = [format_prompt(prompt_file,"What is the smallest multiple of 6 greater than 115?")]
    answers = [example['solution'] for example in examples[:1024]]
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"], include_stop_str_in_output=True
    )
    acc = evaluate_vllm(model, r1_zero_reward_fn, prompts, answers, sampling_params, logger, log=False)
    return acc