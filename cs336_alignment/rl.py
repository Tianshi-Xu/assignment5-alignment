from einops import rearrange
import torch
from typing import Literal
import argparse
from module_rl import *
from utils import *
from dataset import QuestionDataset
import yaml
from timm.utils import setup_default_logging
from logging.handlers import RotatingFileHandler
import logging
import swanlab
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


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
] = "reinforce_with_baseline"
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
    verify_model = init_vllm(args.model_id, device="cuda:3", seed=args.seed, gpu_memory_utilization=0.6)
    train_model = AutoModelForCausalLM.from_pretrained(args.model_id).to("cuda:0")

    optimizer, _ = get_optimizer_scheduler(train_model, 100)
    batch_size = args.train_batch_size
    
    for idx, batch in enumerate(get_loader(train_dataset, batch_size)):
        if idx >= args.n_grpo_steps:
            break
        # load_policy_into_vllm_instance(train_model, verify_model)
        with open("cs336_alignment/prompts/r1_zero.prompt", 'r', encoding='utf-8') as f:
            template = f.read().strip()
        prompts, answers = batch
        prompts = [template.format(question=example) for example in prompts]
        
    
if __name__ == "__main__":
    main()