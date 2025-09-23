from typing import Callable, List, Dict
import json
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(model=model_id,device=device,dtype=torch.bfloat16, enable_prefix_caching=True,gpu_memory_utilization=gpu_memory_utilization)

def format_prompt(prompt_file: str, problem: str) -> str:
    """使用模板格式化问题"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read().strip()

    return template.format(question=problem)

def load_math_dataset(jsonl_file: str) -> List[Dict]:
    """加载 MATH 数据集"""
    examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    logger,
    log: bool = False
) -> None:
    with torch.no_grad():
        outputs = vllm_model.generate(prompts, eval_sampling_params)
    score = 0
    llm_response = {"right_format_right_answer": [], "wrong_format_wrong_answer": [], "wrong_format_right_answer": [], "right_format_wrong_answer": []}
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if i==488:
            logger.info(f"prompt: {prompt}")
            logger.info(f"model output: {generated_text}")
            logger.info(f"reward: {reward}")
        reward = reward_fn(generated_text, answers[i])
        score += reward['reward']
        ans = {"prompt": prompt, "generated_text": generated_text, "answer": answers[i], "reward":reward}
        if reward['format_reward'] == 1.0 and reward['answer_reward'] == 1.0:
            llm_response["right_format_right_answer"].append(ans)
        elif reward['format_reward'] == 0.0 and reward['answer_reward'] == 1.0:
            llm_response["wrong_format_right_answer"].append(ans)
        elif reward['format_reward'] == 1.0 and reward['answer_reward'] == 0.0:
            llm_response["right_format_wrong_answer"].append(ans)
        else:
            llm_response["wrong_format_wrong_answer"].append(ans)
    if log:
        print(f"total problems: {len(outputs)}, score: {score}")
        torch.save(llm_response, "llm_response_base.pt")
    logger.info(f"right_format_right_answer:{len(llm_response['right_format_right_answer'])}")
    logger.info(f"wrong_format_right_answer:{len(llm_response['wrong_format_right_answer'])}")
    logger.info(f"right_format_wrong_answer:{len(llm_response['right_format_wrong_answer'])}")
    logger.info(f"wrong_format_wrong_answer:{len(llm_response['wrong_format_wrong_answer'])}")
    return score/len(outputs)

    
if __name__ == "__main__":
    prompt_file = "cs336_alignment/prompts/r1_zero.prompt"
    jsonl_file = "data/math_hf/valid.jsonl"
    examples = load_math_dataset(jsonl_file)
    prompts = [format_prompt(prompt_file, example['problem']) for example in examples]
    # prompts = [format_prompt(prompt_file,"What is the smallest multiple of 6 greater than 115?")]
    answers = [example['solution'] for example in examples]
    # answers = ["Let $M$ be the smallest multiple of 6 greater than 115. $M$ is both a multiple of 2, which means its units digit must be even, and a multiple of 3, which means the sum of its digits is a multiple of 3.  By the first condition, consider multiples of 2 in increasing order: 116, 118, 120, 122, etc. 116 and 118 are not multiples of 3 (since 1+1+6=8 and 1+1+8=10), but 120 is a multiple of 3. Therefore, $M=\\boxed{120}$."]
    # vllm_model = LLM(model="/opt/pretrained_models/Qwen/Qwen2.5-Math-1.5B")
    vllm_model = init_vllm("/opt/pretrained_models/Qwen/Qwen2.5-Math-1.5B", device="cuda:1", seed=42, gpu_memory_utilization=0.5)
    print("---------end deploy--------------")
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    score = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, answers, sampling_params, device="cuda:0", log=True)
    print(f"score: {score}")