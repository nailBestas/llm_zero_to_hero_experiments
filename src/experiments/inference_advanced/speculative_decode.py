from typing import List, Tuple

import torch
from torch import nn

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.tokenizer_utils import load_tokenizer
from src.inference.generation import load_model_and_tokenizer as load_base


def load_big_and_small(device: str = "cuda") -> Tuple[nn.Module, nn.Module, object]:
    """Büyük modeli checkpoint'ten, küçük modeli sıfırdan yükle."""
    big_model, tokenizer = load_base(device=device)

    vocab_size = tokenizer.get_vocab_size()
    small_model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=128,        # daha küçük
        num_heads=2,
        ff_dim=512,
        num_layers=2,
        max_seq_len=128,
        dropout=0.1,
    ).to(device)

    big_model.eval()
    small_model.eval()
    return big_model, small_model, tokenizer


@torch.no_grad()
def greedy_generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 30, device: str = "cuda") -> str:
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


@torch.no_grad()
def speculative_generate(
    big_model: nn.Module,
    small_model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    draft_steps: int = 4,
    device: str = "cuda",
) -> str:
    """
    Çok basit speculative decoding demo'su:
    - Küçük model, her turda draft_steps adet taslak token üretir.
    - Büyük model aynı adımlar için logit hesaplar ve her adımda taslağı kabul/ret eder.
    - Kabul: taslak token çıktıya eklenir.
    - Ret: büyük modelin greedy seçtiği token eklenir ve tur biter.
    """
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids
    generated = torch.tensor([ids], dtype=torch.long, device=device)

    new_tokens = 0
    while new_tokens < max_new_tokens:
        # 1) Küçük model ile taslak üret
        draft_tokens: List[int] = []
        draft_input = generated.clone()

        for _ in range(draft_steps):
            logits_small = small_model(draft_input)
            next_token_small = torch.argmax(logits_small[:, -1, :], dim=-1)
            next_id = next_token_small.item()
            draft_tokens.append(next_id)

            draft_input = torch.cat(
                [draft_input, next_token_small.view(1, 1)], dim=1
            )

            if new_tokens + len(draft_tokens) >= max_new_tokens:
                break

        # 2) Büyük model ile onaylama
        # Büyük model, mevcut generated + tüm taslakları tek seferde görsün
        big_input = torch.cat(
            [generated, torch.tensor([draft_tokens], dtype=torch.long, device=device)],
            dim=1,
        )
        logits_big = big_model(big_input)

        # Son draft segmentinin her adımındaki argmax'ı kontrol et
        start_pos = generated.size(1)
        accepted = 0

        for i, draft_id in enumerate(draft_tokens):
            pos = start_pos + i
            big_logits_step = logits_big[:, pos - 1, :]  # önceki token sonrası
            big_next = torch.argmax(big_logits_step, dim=-1).item()

            if big_next == draft_id:
                # kabul
                generated = torch.cat(
                    [generated, torch.tensor([[draft_id]], dtype=torch.long, device=device)],
                    dim=1,
                )
                new_tokens += 1
                accepted += 1
                if new_tokens >= max_new_tokens:
                    break
            else:
                # reddet ve büyük modelin token'ını ekle, turu bitir
                generated = torch.cat(
                    [generated, torch.tensor([[big_next]], dtype=torch.long, device=device)],
                    dim=1,
                )
                new_tokens += 1
                break

        if accepted == 0 and len(draft_tokens) == 0:
            # Emniyet için, hiçbir şey üretemezse çık
            break

    return tokenizer.decode(generated[0].tolist())
