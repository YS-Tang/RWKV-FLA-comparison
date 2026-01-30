#!/bin/bash

# RWKV vs FLA-RWKV 性能对比测试脚本
# 用法：修改以下环境变量来选择不同的测试配置

MODEL_TYPE="x070"
N_LAYER="12"
N_EMBD="768"

CTX_LEN="512"
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE

# 清理之前的训练文件
rm "$PROJ_DIR"/rwkv-*0.pth
rm "$PROJ_DIR"/rwkv-71.pth
rm "$PROJ_DIR"/rwkv-final.pth

# 训练超参数
M_BSZ="48" # Batch size，可修改为 16 或 48
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=1
EPOCH_SAVE=10

N_NODE=1
GPU_PER_NODE=1
DS_BUCKET_MB=2

# 测试配置
export RUN_TYPE="fla" # 选择测试类型：rwkv 或 fla
export init_method="share" # fla测试初始化的方式，"share"（使用RWKV权重）或"separate"（使用FLA默认初始化）
export RWKV_JIT_ON="0" # rwkv测试时是否开启jit，"1"开启，"0"关闭

# 运行训练
python train.py --load_model "0" --wandb "" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count 5 --epoch_begin 0 \
 --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB
