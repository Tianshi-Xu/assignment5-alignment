from einops import rearrange
import torch
from typing import Literal
import argparse
from module_rl import *
from module_rl import compute_group_normalized_rewards
from utils import *
from dataset import QuestionDataset, r1_zero_reward_fn, RLDataset
import yaml
from timm.utils import setup_default_logging
from logging.handlers import RotatingFileHandler
import logging
import swanlab
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from module_sft import get_response_log_probs,tokenize_prompt_and_output
import time


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Math RL')
parser.add_argument('--model_id', type=str, default="", help="model id")
parser.add_argument('--train_dir', type=str, default="", help="dir for train dataset")
parser.add_argument('--valid_dir', type=str, default="", help="dir for valid dataset")
parser.add_argument('--train_batch_size', type=int, default=256, help="batch size")
parser.add_argument('--n_grpo_steps', type=int, default=200, help="number of grpo steps")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="learning rate")
parser.add_argument('--advantage_eps', type=float, default=1e-6, help="advantage epsilon")
parser.add_argument('--rollout_batch_size', type=int, default=256, help="rollout batch size")
parser.add_argument('--group_size', type=int, default=1, help="group size")
parser.add_argument('--sampling_temperature', type=float, default=1.0, help='sampling temperature')
parser.add_argument('--sampling_min_tokens', type=int, default=4, help='minimum tokens to sample')
parser.add_argument('--sampling_max_tokens', type=int, default=1024, help='maximum tokens to sample')
parser.add_argument('--epochs_per_rollout_batch', type=int, default=1, help='number of epochs per rollout batch')
parser.add_argument('--gradient_accumulation_steps', type=int, default=128, help='gradient accumulation steps')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.45, help='gpu memory utilization')

parser.add_argument('--seed', type=int, default=3742, help="seed")

parser.add_argument('--log_name', default='none', type=str, help='')
parser.add_argument('--out_path', default='none', type=str, help='path to save output model')
parser.add_argument('--wandb_name', default='cs336_assignment5_rl', type=str, help='wandb name')

loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
] = "no_baseline"
use_std_normalization: bool = True

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

def get_optimizer_scheduler(model: PreTrainedModel,num_training_steps):
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

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
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size
    train_dataset = QuestionDataset(args.train_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    verify_model = init_vllm(args.model_id, device="cuda:0", seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    train_model = AutoModelForCausalLM.from_pretrained(args.model_id).to("cuda:0")
    train_device = train_model.device
    optimizer, scheduler = get_optimizer_scheduler(train_model, args.n_grpo_steps*args.rollout_batch_size/args.train_batch_size)
    batch_size = args.train_batch_size
    best_acc = 0
    t = time.time()
    for step, batch in enumerate(get_loader(train_dataset, n_prompts_per_rollout_batch)):
        if step >= args.n_grpo_steps:
            break
        # load_policy_into_vllm_instance(train_model, verify_model)
        with open("cs336_alignment/prompts/r1_zero.prompt", 'r', encoding='utf-8') as f:
            template = f.read().strip()
        prompts, answers = batch
        prompts = [template.format(question=example) for example in prompts]
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0,min_tokens=args.sampling_min_tokens, max_tokens=args.sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output=True, n=args.group_size, logprobs=1
        )
        outputs = verify_model.generate(prompts, sampling_params)
        rollout_responses = [output.outputs[i].text for output in outputs for i in range(args.group_size)]
        rollout_prompts = [output.prompt for output in outputs for _ in range(args.group_size)]
        repeated_ground_truths = [answer for answer in answers for _ in range(args.group_size)]
        rollout_log_probs = [output.outputs[i].logprobs for output in outputs for i in range(args.group_size)]
        
        assert repeated_ground_truths[0] == repeated_ground_truths[args.group_size-1]
        advantages, raw_rewards, reward_data = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, repeated_ground_truths, args.group_size, args.advantage_eps, use_std_normalization)
        rollout_dataset = RLDataset(rollout_prompt=rollout_prompts, rollout_response=rollout_responses, rollout_advantage=advantages, raw_rewards=raw_rewards)
        _logger.info(f"rollout batch {step} generated, length: {len(rollout_dataset)}")
        t = time.time()
        for i in range(args.epochs_per_rollout_batch):
            for idx, train_batch in enumerate(get_loader(rollout_dataset, micro_train_batch_size)):
                rollout_prompt, rollout_response, rollout_advantage, raw_rewards = train_batch
                tokenized_batch = tokenize_prompt_and_output(rollout_prompt, rollout_response, tokenizer)
                input_ids = tokenized_batch["input_ids"].to(train_model.device)
                labels = tokenized_batch["labels"].to(train_model.device)
                response_mask = tokenized_batch["response_mask"].to(train_model.device)
                rollout_advantage = rollout_advantage.to(train_model.device)
                ret = get_response_log_probs(train_model,input_ids,labels,return_token_entropy=True)
                log_probs, token_entropy = ret["log_probs"], ret["token_entropy"]
                # print(log_probs.shape, response_mask.shape, rollout_advantage.shape)
                loss, metadata = grpo_microbatch_train_step(log_probs, response_mask, args.gradient_accumulation_steps,loss_type=loss_type,raw_rewards=raw_rewards.to(train_model.device), advantages=rollout_advantage.to(train_model.device))
                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    l2_norm = grad_clipping(train_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    _logger.info(f"rl_loss: {loss.item()}, time: {time.time() - t}, l2_norm: {l2_norm}, token_entropy: {token_entropy.mean().item()}, reward_data: {reward_data}")
        load_policy_into_vllm_instance(train_model, verify_model)
        if step % 10 == 0:
            with torch.no_grad():
                acc = evaluate(args,_logger,verify_model, sample_num=1024)
            _logger.info(f"step batch {step} done, eval_acc: {acc}")
            if acc > best_acc:
                best_acc = acc
                train_model.save_pretrained(args.out_path + "/"+ args.log_name)
                tokenizer.save_pretrained(args.out_path + "/"+ args.log_name)
                _logger.info(f"in step {step} new best eval_acc: {acc}, save model to {args.out_path + '/'+ args.log_name}")
        del advantages, raw_rewards, reward_data, rollout_dataset, outputs, rollout_prompts, rollout_responses, repeated_ground_truths, rollout_prompt, rollout_response, rollout_advantage, tokenized_batch, input_ids, labels, response_mask, ret, log_probs, token_entropy, loss, metadata
    train_model = AutoModelForCausalLM.from_pretrained(args.out_path + "/"+ args.log_name).to(train_device)
    load_policy_into_vllm_instance(train_model, verify_model)
    args.valid_dir = "data/math_hf/test.jsonl"
    acc = evaluate(args,_logger,verify_model, sample_num=None)
    _logger.info(f"Best eval_acc: {best_acc}")
    _logger.info(f"training time: {time.time() - t}")
    swanlab.finish()
if __name__ == "__main__":
    main()