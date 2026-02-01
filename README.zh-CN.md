# RWKV vs FLA-RWKV 性能对比

[English](README.md) | 中文

本文档对比了 [RWKV](https://github.com/BlinkDL/RWKV-LM) 基准实现与 [FLA](https://github.com/fla-org/flash-linear-attention) (Flash Linear Attention) 库中 RWKV7 实现的性能差异。

## 测试概述

本次测试对比了两个库中 RWKV7 layer 的实现效果。测试基于 [RWKV-v7/train_temp](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) 目录，对比 RWKV 基准实现中的 [RWKV_Tmix_x070](https://github.com/BlinkDL/RWKV-LM/blob/dc6c3ec05d066489b189e1d2cb13455a2523ccfa/RWKV-v7/train_temp/src/model.py#L76) 和 FLA 实现中的 [RWKV7Attention](https://github.com/fla-org/flash-linear-attention/blob/0044986ae05359ff28924f15d88e5372f51a3368/fla/layers/rwkv7.py#L26)。

测试时将本仓库 `train-temp` 目录中的文件与 RWKV-v7/train_temp 中对应文件进行替换。

## 环境要求

- PyTorch 2.5+
- DeepSpeed
- Transformers
- PyTorch Lightning 1.9.5
- FLA (Flash Linear Attention)

## 测试配置

详细的超参数设置请参考测试脚本 [demo-training-run.sh](./train-temp/demo-training-run.sh)。

主要配置参数：
- 模型类型：x070
- 层数：12
- 嵌入维度：768
- 上下文长度：512
- Batch size：16 / 48
- 学习率：6e-4 → 6e-5
- 数据集：MiniPile

## 测试结果

### 1. 初始化影响
RWKV 和 FLA-RWKV 在源代码中采用了不同的初始化方案。为了公平对比，我们分别测试了各自默认初始化方案以及使用相同初始权重的情况。相同的初始权重采用 RWKV 初始化时生成的权重，两个实现的模型命名对齐参见 [RWKV与FLA-RWKV命名对齐](RWKV与FLA-RWKV命名对齐.md)。

测试的超参数设置参见测试脚本 [demo-training-run.sh](./train-temp/demo-training-run.sh)，测试了 batch size 为 48 的场景。

下图展示了训练 2000 步的测试结果。在训练的最后 200 步中，RWKV 与使用同样初始化权重的 FLA 的平均损失值分别为 2.965 和 3.018，RWKV 显著更好；而采用默认初始化的 FLA 损失值为 3.670，表现很差。

<img src="./figures/Training loss with error band (bsz=48).png" alt="Training loss comparison (bsz=48)"/>

**结论**：初始化影响显著：FLA-RWKV 使用默认初始化时性能很差，使用 RWKV 基准实现的初始化权重后仍显著低于 RWKV。

### 2. 速度对比
RWKV 采用了自定义的 CUDA 和 C++ 加速方案来提升处理速度，并支持 JIT（即时编译）优化。相比之下，FLA-RWKV 由于使用了 lambda 函数，不便使用 JIT 加速。为了对比速度差异，我们分别测试了开启和关闭 JIT 的 RWKV 以及 FLA-RWKV。

下图展示了 batch size 为 48 时的每秒千词元数（kt/s）。在使用 IQR 方法剔除极端数据后，关闭和开启 JIT 情况下的 RWKV 速度分别为 41.443 kt/s 和 46.253 kt/s，而 FLA-RWKV 的速度为 34.365 kt/s。

<img src="./figures/Kilo-tokens per second (bsz=48).png" alt="Training speed comparison (bsz=48)"/>

**结论**：RWKV 基准实现在训练速度上明显优于 FLA-RWKV, FLA-RWKV 的速度约为 RWKV 的 83% 左右。开启 JIT 后，RWKV 的速度再提升约 11-12%。

## 总结

1. **初始化影响显著**：FLA-RWKV 使用默认初始化时性能很差，使用 RWKV 基准实现的初始化权重后仍显著低于 RWKV。
2. **训练速度差异**：RWKV 基准实现在训练速度上明显优于 FLA-RWKV, FLA-RWKV 的速度约为 RWKV 的 83% 左右。开启 JIT 后，RWKV 的速度再提升约 11-12%。
