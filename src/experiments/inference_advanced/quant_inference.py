import time
from typing import Tuple

import torch
from torch import nn

from src.inference.generation import load_model_and_tokenizer as load_base


@torch.no_grad()
def greedy_generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 30, device: str = "cuda") -> Tuple[str, float]:
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    t1 = time.time()

    text = tokenizer.decode(input_ids[0].tolist())
    return text, t1 - t0


def load_fp32(device: str = "cuda"):
    model, tokenizer = load_base(device=device)
    model.eval()
    return model, tokenizer


def load_fp16(device: str = "cuda"):
    model, tokenizer = load_base(device=device)
    model = model.half()  # ağırlıkları ve hesaplamayı fp16 yap
    model.eval()
    return model, tokenizer
