#!/bin/bash

NUM_GPUS = $(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NUM_GPUS train.py --model_size 1B --block_pattern MDMA --attn_type nsa --optimizer_type adamw8bit -lr 1e-3 -wd 0.05 -clip 2.0
torchrun --nproc_per_node=$NUM_GPUS train.py --model_size 1B --block_pattern MDMA --attn_type nsa --optimizer_type soap8bit  -lr 1e-3 -wd 0.05 -clip 2.0 
torchrun --nproc_per_node=$NUM_GPUS train.py --model_size 1B --block_pattern MDMA --attn_type nsa --optimizer_type soap4bit  -lr 1e-3 -wd 0.05 -clip 2.0
