#!/usr/bin/env bash
export PYTHONPATH=.

# GPU sayına göre ayarla (ör: 2)
WORLD_SIZE=2

torchrun --nproc_per_node=$WORLD_SIZE src/experiments/train_scale/train_fsdp.py
