#!/usr/bin/env bash
export PYTHONPATH=.

WORLD_SIZE=1   # makinede 1 GPU var

torchrun --nproc_per_node=$WORLD_SIZE src/experiments/train_scale/train_fsdp.py
