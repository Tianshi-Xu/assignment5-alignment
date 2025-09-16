import json
from typing import List, Dict
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    create_dataset("data/am/am_cs336.jsonl")