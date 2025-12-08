#!/usr/bin/env bash
export PYTHONPATH=.

deepspeed src/experiments/train_scale/train_deepspeed.py

