from transformers import PreTrainedTokenizer, AutoTokenizer
from einops import rearrange
import torch
from torch.utils.data import Dataset
import json
import os
from typing import BinaryIO, IO

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False).input_ids
    output_ids = tokenizer(output_strs, add_special_tokens=False).input_ids
    input_ids = [p + o for p, o in zip(prompt_ids, output_ids)]
    # 3. label_mask: prompt部分=0, output部分=1
    # 4. padding (对 input_ids 和 label_masks 一起 pad)
    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )
    label_masks = torch.tensor([[0] * len(p) + [1] * (len(k)-len(p)) for p, k in zip(prompt_ids,batch["attention_mask"])])
    label_masks = label_masks * batch["attention_mask"]
    input_ids = batch["input_ids"][:,:-1]
    labels = batch["input_ids"][:,1:]
    label_masks = label_masks[:,1:]
    ans = {"input_ids": input_ids, "labels": labels, "response_mask": label_masks}
    return ans
    
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    tmp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(torch.exp(tmp)/torch.sum(torch.exp(tmp), dim=-1, keepdim=True) * tmp, dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = log_probs.gather(dim=-1, index=rearrange(labels, 'b s -> b s 1')).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs":log_probs, "token_entropy":token_entropy}
    else:
        return {"log_probs":log_probs}
    
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.sum(tensor * mask, dim=dim) / normalize_constant
 
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    loss = -masked_normalize(policy_log_probs, response_mask, dim=1, normalize_constant=normalize_constant)
    loss = torch.mean(loss)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return (loss, {})

def log_generations(
    input_prompt: str,
    response: str,
    ground_truth: str,
    reward: dict,    
) -> dict:
    pass

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    state_dict = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"iteration":iteration}
    torch.save(state_dict,out)

class SFTDataset(Dataset):
    def __init__(self, jsonl_file: str):
        self.prompt = []
        self.response = []
        self.max_length = 4096
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_line = json.loads(line)
                    self.prompt.append(json_line["prompt"])
                    self.response.append(json_line["response"])

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx) -> tuple[str, str]:
        return self.prompt[idx], self.response[idx]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/opt/pretrained_models/Qwen/Qwen2.5-Math-1.5B/")
    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    print(tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer))