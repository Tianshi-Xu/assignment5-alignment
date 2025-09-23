from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from unittest.mock import patch
import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from typing import Callable, List, Dict
import json
import argparse
import logging,yaml
from logging.handlers import RotatingFileHandler
from timm.utils import setup_default_logging
import swanlab
from cs336_alignment.module_sft import *
from eval_math import load_math_dataset, format_prompt, evaluate_vllm, r1_zero_reward_fn
import time

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Math SFT')
parser.add_argument('--model_id', type=str, default="", help="model id")
parser.add_argument('--train_dir', type=str, default="", help="dir for train dataset")
parser.add_argument('--valid_dir', type=str, default="", help="dir for valid dataset")
parser.add_argument('--num_samples', type=int, default=128, help="number of samples")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation steps")
parser.add_argument('--seed', type=int, default=3742, help="seed")

parser.add_argument('--log_name', default='none', type=str, help='')
parser.add_argument('--out_path', default='none', type=str, help='path to save output model')
parser.add_argument('--wandb_name', default='cs336_assignment5_sft', type=str, help='wandb name')

_logger = logging.getLogger('train')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

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


def create_dataset(jsonl_file: str) -> List[Dict]:
    """加载 MATH 数据集"""
    examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    new_datasets = []
    for example in examples:
        new_dataset = {
            "prompt": example["messages"][0]["content"],
            "response": example["messages"][1]["content"]
        }
        new_datasets.append(new_dataset)
    with open("data/am/am_cs336.jsonl", "w", encoding="utf-8") as f:
        for item in new_datasets:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return new_datasets

def get_loader(dataset: List[Dict], batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_optimizer_scheduler(model: PreTrainedModel,num_training_steps):
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def evaluate(args,logger,model:LLM) -> float:
    prompt_file = "cs336_alignment/prompts/r1_zero.prompt"
    jsonl_file = args.valid_dir
    examples = load_math_dataset(jsonl_file)
    prompts = [format_prompt(prompt_file, example['problem']) for example in examples]
    # prompts = [format_prompt(prompt_file,"What is the smallest multiple of 6 greater than 115?")]
    answers = [example['solution'] for example in examples]
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"], include_stop_str_in_output=True
    )
    acc = evaluate_vllm(model, r1_zero_reward_fn, prompts, answers, sampling_params, logger, log=False)
    return acc

def main():
    g = torch.Generator()
    g.manual_seed(torch.seed())
    args, args_text = _parse_args()
    setup_default_logging()
    handler = RotatingFileHandler(args.log_name+'.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    swanlab.init(project=args.wandb_name, config=args, name=args.log_name)
    train_dataset = SFTDataset(is_jsonl=True, jsonl_file=args.train_dir)
    # valid_dataset = SFTDataset(args.valid_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # special_tokens = {"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]}
    # tokenizer.add_special_tokens(special_tokens)
    # print(tokenizer.tokenize("<think>, </think>, <answer>, </answer>"))
    # exit(0)
    verify_model = init_vllm(args.model_id, device="cuda:1", seed=args.seed, gpu_memory_utilization=0.4)
    train_model = AutoModelForCausalLM.from_pretrained(args.model_id).to("cuda:0")
    # train_model.resize_token_embeddings(len(tokenizer))
    # get optimizer and scheduler
    num_training_steps = args.num_samples // args.gradient_accumulation_steps

    optimizer, scheduler = get_optimizer_scheduler(train_model, num_training_steps)
    # load_policy_into_vllm_instance(train_model, verify_model)
    # batch load dataset
    batch_size = args.batch_size
    # after each 1/4, evaluate the model
    eval_interval = args.num_samples//4
    _logger.info(f"eval_interval: {eval_interval}")
    best_loss = 1e9
    best_acc = 0
    load_policy_into_vllm_instance(train_model, verify_model)
    acc = evaluate(args,_logger,verify_model)
    _logger.info(f"Initial eval_acc: {acc}")
    t = time.time()
    for idx, batch in enumerate(get_loader(train_dataset, batch_size)):
        if idx >= args.num_samples:
            break
        prompt, response = batch
        tokenized_batch = tokenize_prompt_and_output(prompt, response, tokenizer)
        input_ids = tokenized_batch["input_ids"].to(train_model.device)
        labels = tokenized_batch["labels"].to(train_model.device)
        response_mask = tokenized_batch["response_mask"].to(train_model.device)
        
        ret = get_response_log_probs(train_model,input_ids,labels,return_token_entropy=True)
        log_probs, token_entropy = ret["log_probs"], ret["token_entropy"]
        loss, _ = sft_microbatch_train_step(log_probs, response_mask, args.gradient_accumulation_steps)
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            _logger.info(f"Step {idx} loss: {loss.item()} token_entropy: {token_entropy.mean().item()}")
            swanlab.log({"train_loss":loss.item(),"train_token_entropy":token_entropy.mean().item(), "time":time.time() - t})
            if loss.item() < best_loss:
                best_loss = loss.item()
                # if idx % 10 == 0:
                #     run_save_checkpoint(train_model, optimizer, idx, args.out_path + "/"+ args.log_name + "_best_model.pt")
        if (idx + 1) % eval_interval == 0:
            eval_interval += args.num_samples//4
            # print("begin eval")
            load_policy_into_vllm_instance(train_model, verify_model)
            acc = evaluate(args,_logger,verify_model)
            swanlab.log({"eval_acc":acc, "time":time.time() - t})
            _logger.info(f"Step {idx} eval_acc: {acc}")
            if acc > best_acc:
                best_acc = acc
                train_model.save_pretrained(args.out_path + "/"+ args.log_name)
                tokenizer.save_pretrained(args.out_path + "/"+ args.log_name)
    train_model = AutoModelForCausalLM.from_pretrained(args.out_path + "/"+ args.log_name).to("cuda:2")
    load_policy_into_vllm_instance(train_model, verify_model)
    args.valid_dir = "data/math_hf/test.jsonl"
    acc = evaluate(args,_logger,verify_model)
    _logger.info(f"Final eval_acc: {acc}")
    _logger.info(f"Best loss: {best_loss}")
    _logger.info(f"Best acc: {best_acc}")
    _logger.info(f"Training time: {(time.time()-t)/60} min")
    swanlab.finish()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    