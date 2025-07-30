# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# R1 template
# export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/orion/orion_runtime/gpu/cuda/orion-cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# export NCCL_P2P_DISABLE=1  # Sometimes helps with multi-GPU issues
# export CUDA_LAUNCH_BLOCKING=1  # For better error messages during debugging
# export WANDB_API_KEY="47d96e0a83c4787dd7fd11acd2d899108c98de97"
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="/opt/conda/envs/oat09/bin:$PATH"

#export NCCL_SOCKET_IFNAME=lo
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export NCCL_SOCKET_FAMILY=AF_INET
#export NCCL_DEBUG=INFO
#export NCCL_CUMEM_ENABLE=0
#export NCCL_P2P_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
# export WANDB_MODE=offline

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

p1=0
p2=0
num_samples=8
max_tries=1
model_type=1
#reward_noise_p=0
# 0 means base
# 1 means 1.5B-instruct \ 7B-math

# --- 逻辑结束 ---
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Found ${num_gpus} GPUs in CUDA_VISIBLE_DEVICES. Setting --gpus=${num_gpus}"

p1_str=$(printf "%.2f" $p1)  # 格式化为两位小数，确保稳定性
p2_str=$(printf "%.2f" $p2)  # 格式化为两位小数，确保稳定性


# 构造 log_path 和 run_name
#log_path="log/grpotest-dhgrpo-qwen2.5-math-1.5b-ins-p-${p_str}-p2-0.01"
#run_name="grpotest-dhgrpo-qwen2.5-math-1.5b-ins-p-${p_str}-p2-0.01"

log_path="log2/1.5B-p1-${p1_str}-p2-${p2_str}-num_samples-${num_samples}-qwen-072418"
#log_path="log2/1.5b-p1-${p1_str}-p2-${p2_str}-num_samples-${num_samples}-sample-${max_tries}try-num_gpus-${num_gpus}-1920"

run_name="1.5B-p1-${p1_str}-p2-${p2_str}-num_samples-${num_samples}-qwen-072418"

mkdir -p "$log_path"
#/gemini/data-1/Qwen/Qwen2___5-Math-1___5B-Instruct/ /gemini/data-1/Qwen/Qwen2.5-Math-7B/  r1
#    --reward_noise_p ${reward_noise_p} \
PYTHONUNBUFFERED=1 nohup stdbuf -oL -eL python train_zero_math.py\
    --wb-run-name "${run_name}" \
    --max_tries ${max_tries} \
    --critic_type drgrpo \
    --drgrpo_p1 ${p1} \
    --drgrpo_p2 ${p2} \
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
    --pretrain  /gemini/data-1/Qwen/Qwen2___5-Math-1___5B-Instruct/ \
    --prompt_template qwen_math \
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
    --top_p 1 \
    --generate_max_length 4096 \
    --save_steps 50 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device $((128 / num_gpus)) \
    --pi_buffer_maxlen_per_device $((128 * num_samples / num_gpus)) \
    --eval_batch_size 200 \
    --eval_steps 5 \
    --eval_temperature 0 \
    --eval_generate_max_length 4096 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input  \
    > "$log_path/run.log" 2>&1 & echo $! > "$log_path/run.pid"


# Qwen-Math template
# python train_zero_math.py \
#     --critic_type drgrpo \
#     --gpus 8 \
#     --enable_prefix_caching \
#     --collocate \
#     --vllm_sleep \
#     --vllm_gpu_ratio 0.35 \
#     --gradient-checkpointing \
#     --flash-attn \
#     --bf16 \
#     --rnd-seed \
#     --learning_rate 0.000001 \
#     --lr_scheduler constant \
#     --num_ppo_epochs 1 \
#     --beta 0 \
#     --oracle_type reward \
#     --oracle math \
#     --pretrain Qwen/Qwen2.5-Math-7B \
#     --prompt_template qwen_math \
#     --verifier_version math_verify \
#     --zero-stage 2 \
#     --ref_offload \
#     --prompt_data ./datasets/train/math_lvl3to5_8k \
#     --train_split train \
#     --input_key problem \
#     --output_key answer \
#     --max-train 9999999 \
#     --num_prompt_epoch 20 \
#     --prompt_max_length 1024 \
#     --num_samples 8 \
#     --temperature 1 \
#     --top_p 1 \
#     --generate_max_length 3000 \
#     --save_steps -1 \
#     --train_batch_size 128 \
#     --train_batch_size_per_device 1 \
#     --mini_train_batch_size_per_device 1 \
#     --rollout_batch_size 128 \
#     --rollout_batch_size_per_device 16 \
#     --pi_buffer_maxlen_per_device 128 \
#     --eval_batch_size 200 \
#     --eval_steps 16 \
#     --eval_temperature 0 \
#     --eval_generate_max_length 3000 \
#     --eval_data ./datasets/evaluation_suite \
#     --eval_input_key input \
#     --use-wb \
#     --wb_project oat-zero \
#     --wb-run-name qwen2.5-Math-7b-drgrpo-qwenmathtemplate
