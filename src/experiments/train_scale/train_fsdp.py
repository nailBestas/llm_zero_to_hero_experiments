import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP  # sadece rank kontrolü için
from torch.utils.data import DataLoader

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.dataset import TextLineDataset
from src.data_pipeline.tokenizer_utils import load_tokenizer


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def create_dataloader(files, tokenizer_path, max_seq_len=128, batch_size=8, num_workers=2):
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = TextLineDataset(files=files, tokenizer=tokenizer, max_seq_len=max_seq_len)
    # FSDP ile genelde DistributedSampler kullanılır ama basitlik için şimdilik düz DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader, tokenizer


def create_fsdp_model(vocab_size: int, device: torch.device) -> FSDP:
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        ff_dim=1024,
        num_layers=6,   # DDP örneğinden biraz daha derin; FSDP kazancını görmek için
        max_seq_len=128,
        dropout=0.1,
    ).to(device)

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,
    )
    return fsdp_model


def train_one_epoch(model: FSDP, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device, rank: int):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if rank == 0 and (step + 1) % 100 == 0:
            avg = total_loss / total_batches
            print(f"[rank 0] step {step+1}: loss={avg:.4f}")

    return total_loss / max(total_batches, 1)


def save_fsdp_checkpoint(model: FSDP, optimizer: torch.optim.Optimizer, path: str, rank: int):
    if rank != 0:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Basit tam state_dict kaydı (daha gelişmiş FSDP checkpoint API'leri de var)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"[rank 0] Saved FSDP checkpoint to {path}")


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"World size (GPUs): {world_size}")

    train_files = ["data/raw/train.txt"]   # kendi dosya yoluna göre ayarla
    tokenizer_path = "data/tokenizer/tokenizer.json"
    max_seq_len = 128
    batch_size = 8
    lr = 3e-4
    epochs = 1

    dataloader, tokenizer = create_dataloader(
        files=train_files,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=2,
    )

    vocab_size = tokenizer.get_vocab_size()
    model = create_fsdp_model(vocab_size=vocab_size, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, rank)
        if rank == 0:
            print(f"[rank 0] Epoch {epoch}/{epochs}: avg_loss={avg_loss:.4f}")

    save_fsdp_checkpoint(model, optimizer, "models/mini_lm_fsdp/checkpoint.pt", rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
