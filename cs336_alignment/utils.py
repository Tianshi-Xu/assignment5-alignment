from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from unittest.mock import patch
import torch
from vllm import LLM
from torch.utils.data import DataLoader
from typing import List, Dict


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(model=model_id,device=device,dtype=torch.bfloat16, enable_prefix_caching=True,gpu_memory_utilization=gpu_memory_utilization)


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
def get_loader(dataset: List[Dict], batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)