#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT

n_clients=${1:-1}
split=${2:-iid}
sigma=${3:-0}
local_epoch=${4:-10}
shuffle_percentage=${5:-0.0}
method=${6:-check_zeta}
version=${7:-0}


gpu_index=${8:-1}
s_lr=${9:-0}
num_rounds=${10:-50}

export CUDA_VISIBLE_DEVICES="$gpu_index"
python3 train_mnist.py --n_clients "$n_clients" --split "$split" --sigma "$sigma" --num_local_epochs "$local_epoch" \
    --shuffle_percentage 0 --method "$method" --version "$version" \
    --lr "$s_lr" \
    --num_rounds "$num_rounds" 
