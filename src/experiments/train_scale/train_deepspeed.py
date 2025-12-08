import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import deepspeed

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.dataset import TextLineDataset
from src.data_pipeline.tokenizer_utils import load_tokenizer

DS_CONFIG = {
    "train_batch_size": 8,              # 32 yerine daha küçük
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e7,      # 50M param civarı (~200MB civarı buffer)
        "allgather_bucket_size": 5e7
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-4}
    },
}





def create_dataloader(files, tokenizer_path, max_seq_len=128, batch_size=8, num_workers=2):
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = TextLineDataset(files=files, tokenizer=tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader, tokenizer


def create_model(vocab_size: int) -> MiniTransformerLM:
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        ff_dim=1024,
        num_layers=6,
        max_seq_len=128,
        dropout=0.1,
    )
    return model


def train_one_epoch(engine, dataloader, rank: int):
    engine.train()
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(engine.device)
        labels = batch["labels"].to(engine.device)

        outputs = engine(input_ids)
        logits = outputs  # MiniTransformerLM doğrudan logits döndürüyor
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        engine.backward(loss)
        engine.step()

        total_loss += loss.item()
        total_batches += 1

        if rank == 0 and (step + 1) % 100 == 0:
            avg = total_loss / total_batches
            print(f"[rank 0] step {step+1}: loss={avg:.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(engine, tokenizer, output_dir: str, rank: int):
    if rank != 0:
        return
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # DeepSpeed engine, .module üzerinden asıl modeli tutar
    state = {
        "model_state_dict": engine.module.state_dict(),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
    }
    ckpt_path = out_path / "checkpoint_deepspeed.pt"
    torch.save(state, ckpt_path)
    print(f"[rank 0] Saved DeepSpeed checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()


    # DeepSpeed launcher rank/env değişkenlerini ayarlar
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print(f"DeepSpeed world size: {world_size}")

    train_files = ["data/raw/train.txt"]  # kendi dosyana göre güncelle
    tokenizer_path = "data/tokenizer/tokenizer.json"
    max_seq_len = 128
    batch_size = 8
    epochs = 1

    dataloader, tokenizer = create_dataloader(
        files=train_files,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=2,
    )

    vocab_size = tokenizer.get_vocab_size()
    model = create_model(vocab_size=vocab_size)

    # DeepSpeed engine initialize
    engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=DS_CONFIG,
    )


    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(engine, dataloader, rank)
        if rank == 0:
            print(f"[rank 0] Epoch {epoch}/{epochs}: avg_loss={avg_loss:.4f}")

    save_checkpoint(engine, tokenizer, "models/mini_lm_deepspeed", rank)


if __name__ == "__main__":
    main()
