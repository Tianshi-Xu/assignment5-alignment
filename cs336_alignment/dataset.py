import json
from typing import List, Dict
import matplotlib.pyplot as plt
import re
from drgrpo_grader import r1_zero_reward_fn
from torch.utils.data import Dataset

def create_dataset(jsonl_file: str) -> List[Dict]:
    """加载 MATH 数据集"""
    sta_len = []
    number = 2048
    sta_less = 0
    less_dataset = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sta_len.append(len(data["prompt"])+len(data["response"]))
                if len(data["prompt"])+len(data["response"]) < number:
                    sta_less += 1
                    less_dataset.append(data)
    print(f"mean: {sum(sta_len)/len(sta_len)}")
    print(f"less than {number}: {sta_less}")
    print(f"total len:{len(sta_len)}")
    # draw the distribution of sta_len
    plt.hist(sta_len, bins=100)
    plt.savefig("sta_len.png")
    with open(f"data/am/am_cs336_less{number}.jsonl", "w", encoding="utf-8") as f:
        for item in less_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return sta_len
# {"prompt_index": 0, "prompt": "Below is a math question, first thinks about the reasoning process in the mind and then provides with the answer. The reasoning process is enclosed within <mythink> </mythink> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <mythink> reasoning process here </mythink> <answer> answer here </answer>. Question: Determine the average of the integers $71,$ $72,$ $73,$ $74,$ $75.$", "response": "<mythink>\nThe average of a set of numbers is calculated by summing all the numbers and then dividing by the count of numbers. The given integers are 71, 72, 73, 74, and 75.\n\nFirst, find the sum:\n71 + 72 = 143  \n143 + 73 = 216  \n216 + 74 = 290  \n290 + 75 = 365  \n\nThe sum is 365.\n\nThere are 5 numbers in the set.\n\nNow, divide the sum by the count:  \n365 ÷ 5 = 73.\n\nAlternatively, since the numbers are consecutive integers and there is an odd count (5), the average is the middle number, which is 73. This confirms the result.\n\nThus, the average is 73.\n</mythink>\n\n<answer>73</answer>", "success": true, "error": null}
def process_data(jsonl_file: str, correct=False) -> List[Dict]:
    new_dataset = []
    prompt_file = "cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read().strip()
    if correct:
        examples = []
        with open("data/math_hf/train.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data["success"]:
                    # 提取 Question: 后面的问题
                    prompt = data["prompt"]
                    question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', prompt, re.DOTALL)
                    if question_match:
                        problem = question_match.group(1).strip()
                    else:
                        # 如果没有找到Question:，跳过这条数据
                        continue
                    ## if </think>\n\n<answer> not in response, skip
                    if "</mythink>\n\n<answer>" not in data["response"]:
                        # print(data["response"])
                        continue
                    # 使用新模板替换prompt
                    new_prompt = template.format(question=problem)
                    
                    # 替换response中的<mythink>标签为<think>
                    response = data["response"]
                    # if don't have <mythink>, skip
                    if "<mythink>" not in response or "</mythink>" not in response:
                        continue
                    # print(data["prompt_index"])
                    response = response.replace("<mythink>", "<think>")
                    if correct:
                        solution = examples[data["prompt_index"]]["solution"]
                        reward_fn = r1_zero_reward_fn(response.replace("</mythink>", "</think>"), solution)
                        if reward_fn["reward"] == 0:
                            continue
                    ### delete all content before <think>, but need to keep <think>
                    response = response.split("<think>")[1]
                    # response = response.replace("<think>", "")
                    response = response.replace("</mythink>", "</think>")
                    
                    # 创建新的数据项
                    new_data = {
                        "prompt_index": data.get("prompt_index", 0),
                        "prompt": new_prompt,
                        "response": response,
                    }
                    new_dataset.append(new_data)
    print("valid total len:", len(new_dataset))
    with open(f"data/math_hf/sft_correct.jsonl" if correct else "data/math_hf/sft.jsonl", "w", encoding="utf-8") as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return new_dataset

class SFTDataset(Dataset):
    def __init__(self, is_jsonl: bool, jsonl_file: str = None, prompt: list[str] = None, response: list[str] = None):
        self.prompt = []
        self.response = []
        self.max_length = 4096
        
        if is_jsonl:
            # Initialize from JSONL file
            if jsonl_file is None:
                raise ValueError("jsonl_file must be provided when is_jsonl=True")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        json_line = json.loads(line)
                        self.prompt.append(json_line["prompt"])
                        self.response.append(json_line["response"])
        else:
            # Initialize from lists
            if prompt is None or response is None:
                raise ValueError("Both prompt and response must be provided when is_jsonl=False")
            if len(prompt) != len(response):
                raise ValueError("prompt and response lists must have the same length")
            self.prompt = prompt
            self.response = response
    
    @classmethod           
    def from_dataset(cls, dataset: list[dict]):
        prompt = [item["prompt"] for item in dataset]
        response = [item["response"] for item in dataset]
        return cls(is_jsonl=False, jsonl_file= None, prompt=prompt, response=response)

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx) -> tuple[str, str]:
        return self.prompt[idx], self.response[idx]

class RLDataset(Dataset):

    def __init__(self, rollout_prompt: list[str] = None, rollout_response: list[str] = None, rollout_advantage: list[float] = None, raw_rewards: list[float] = None, old_log_probs: list[float] = None):
        self.rollout_prompt = []
        self.rollout_response = []
        self.rollout_advantage = []
        self.raw_rewards = []
        self.old_log_probs = []
        self.max_length = 4096
        self.rollout_prompt = rollout_prompt
        self.rollout_response = rollout_response
        self.rollout_advantage = rollout_advantage
        self.old_log_probs = old_log_probs
        self.raw_rewards = raw_rewards

    def __len__(self):
        return len(self.rollout_prompt)

    def __getitem__(self, idx) -> tuple[str, str]:
        return self.rollout_prompt[idx], self.rollout_response[idx], self.rollout_advantage[idx], self.old_log_probs[idx], self.raw_rewards[idx]


class QuestionDataset(Dataset):
    def __init__(self, jsonl_file: str):
        self.problem = []
        self.answer = []
        self.max_length = 4096
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_line = json.loads(line)
                    self.problem.append(json_line["problem"])
                    self.answer.append(json_line["solution"])

    def __len__(self):
        return len(self.problem)

    def __getitem__(self, idx) -> tuple[str, str]:
        return self.problem[idx], self.answer[idx]

if __name__ == "__main__":
    process_data("data/math_hf/train_math_responses.jsonl", correct=True)