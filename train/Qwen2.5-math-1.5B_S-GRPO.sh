#!/bin/bash

# Set the library path for CUDA
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Set the path for the conda environment
export PATH="/opt/conda/envs/oat09/bin:$PATH"

# Optional NCCL environment variables for network configuration
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0

# Specify the CUDA devices to be used
export CUDA_VISIBLE_DEVICES=0,1,2,3


# --- Parameter Configuration ---
# p: The noise value from the paper.
p=0.15
# num_samples: The number of samples to generate.
num_samples=8
# max_tries: Set to 1 as it is not meaningful in this context.
max_tries=1
# --- End of Configuration ---

# Calculate the number of GPUs being used
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Found ${num_gpus} GPUs in CUDA_VISIBLE_DEVICES. Setting --gpus=${num_gpus}"

# Format p to two decimal places for consistent naming
p_str=$(printf "%.2f" $p)

# Construct the log_path and run_name for experiment tracking
log_path="log_path"
run_name="run_name"
mkdir -p "$log_path"

# Execute the training script using nohup for background execution
PYTHONUNBUFFERED=1 nohup stdbuf -oL -eL python train_S-GRPO.py \
    --wb-run-name "${run_name}" \
    --max_tries ${max_tries} \
    --critic_type drgrpo \
    --drgrpo_p ${p} \
    --gpus ${num_gpus} \
    --enable_prefix_caching \
    --vllm-sleep \
    --collocate \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward\
    --oracle math \
    --pretrain ./Qwen2___5-Math-1___5B-Instruct/ \
    --prompt_template r1 \
    --verifier_version math_verify \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data ./datasets/train/math_lvl3to5_8k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 20 \
    --prompt_max_length 1024 \
    --num_samples ${num_samples} \
    --temperature 1 \
    --top_p 0.95 \
    --generate_max_length 4096 \
    --save_steps 100 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device $((128 / num_gpus)) \
    --pi_buffer_maxlen_per_device $((128 * num_samples / num_gpus)) \
    --eval_batch_size 200 \
    --eval_steps 16 \
    --eval_temperature 0 \
    --eval_generate_max_length 4096 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input \
    > "$log_path/run.log" 2>&1 & echo $! > "$log_path/run.pid"
