from typing import List

import torch
from torch import nn

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.tokenizer_utils import load_tokenizer
from src.inference.generation import load_model_and_tokenizer as load_base


@torch.no_grad()
def naive_generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 30, device: str = "cuda") -> str:
    model.eval()
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids  # <-- BURASI ÖNEMLİ
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)



    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


@torch.no_grad()
def kv_cache_generate_dummy(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 30, device: str = "cuda") -> str:
    """
    Dummy KV cache benzeri generate:
    - Şu an model içinde gerçek K/V cache API'si yok.
    - Bu fonksiyon sadece 'tek token ekleyerek' ilerleme mantığını gösterir.
    - Performans kazancı sağlamaz ama incremental decoding akışını taklit eder.
    """
    model.eval()
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Prefill: tüm prompt'u ver
    logits = model(input_ids)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Sanki sadece son token ile çalışıyormuşuz gibi davranıyoruz
        last_token = generated[:, -1:].to(device)
        logits_step = model(generated)  # gerçek KV cache olsaydı sadece last_token ile çağırırdık
        next_token = torch.argmax(logits_step[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0].tolist())


def load_experiment_model(device: str = "cuda"):
    """Experiments projesindeki mevcut inference loader'ı kullan."""
    model, tokenizer = load_base(device=device)
    return model, tokenizer
