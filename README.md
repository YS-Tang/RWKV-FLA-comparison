# RWKV vs FLA-RWKV Performance Comparison

English | [中文](README.zh-CN.md)

This document compares the performance differences between the [RWKV](https://github.com/BlinkDL/RWKV-LM) baseline implementation and the RWKV7 implementation in the [FLA](https://github.com/fla-org/flash-linear-attention) (Flash Linear Attention) library.

## Test Overview

This test compares the implementation effects of RWKV7 layers in both libraries. The test is based on the [RWKV-v7/train_temp](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) directory, comparing [RWKV_Tmix_x070](https://github.com/BlinkDL/RWKV-LM/blob/dc6c3ec05d066489b189e1d2cb13455a2523ccfa/RWKV-v7/train_temp/src/model.py#L76) in the RWKV baseline implementation and [RWKV7Attention](https://github.com/fla-org/flash-linear-attention/blob/0044986ae05359ff28924f15d88e5372f51a3368/fla/layers/rwkv7.py#L26) in the FLA implementation.

During testing, files in the `train-temp` directory of this repository are replaced with corresponding files in RWKV-v7/train_temp.

## Requirements

- PyTorch 2.5+
- DeepSpeed
- Transformers
- PyTorch Lightning 1.9.5
- FLA (Flash Linear Attention)

## Test Configuration

For detailed hyperparameter settings, please refer to the test script [demo-training-run.sh](./train-temp/demo-training-run.sh).

Main configuration parameters:
- Model type: x070
- Number of layers: 12
- Embedding dimension: 768
- Context length: 512
- Batch size: 16 / 48
- Learning rate: 6e-4 → 6e-5
- Dataset: MiniPile

## Test Results

### 1. Initialization Impact

RWKV and FLA-RWKV adopt different initialization schemes in their source code. For a fair comparison, we tested both their default initialization schemes and the case using the same initial weights. The same initial weights use the weights generated during RWKV initialization. For model naming alignment between the two implementations, see [RWKV与FLA-RWKV命名对齐](RWKV与FLA-RWKV命名对齐.md).

The hyperparameter settings for the test can be found in the test script [demo-training-run.sh](./train-temp/demo-training-run.sh), which tested scenarios with a batch size of 48.

The figure below shows the test results for 2000 training steps. In the final 200 steps of training, the average loss values for RWKV and FLA using the same initialization weights were 2.965 and 3.018 respectively, with RWKV performing significantly better; while FLA with default initialization had a loss value of 3.670, performing poorly.

<img src="./figures/Training loss with error band (bsz=48).png" alt="Training loss comparison (bsz=48)"/>

**Conclusion**: Initialization has a significant impact: FLA-RWKV performs poorly with default initialization, and even after using the initialization weights from the RWKV baseline implementation, it still performs significantly worse than RWKV.

### 2. Speed Comparison

RWKV uses custom CUDA and C++ acceleration schemes to improve processing speed and supports JIT (Just-In-Time) optimization. In contrast, FLA-RWKV cannot use JIT acceleration conveniently due to the use of lambda functions. To compare speed differences, we tested RWKV with JIT enabled and disabled, as well as FLA-RWKV.

The figure below shows the kilo-tokens per second (kt/s) when the batch size is 48. After removing extreme data using the IQR method, the speeds of RWKV with JIT disabled and enabled were 41.443 kt/s and 46.253 kt/s respectively, while the speed of FLA-RWKV was 34.365 kt/s.

<img src="./figures/Kilo-tokens per second (bsz=48).png" alt="Training speed comparison (bsz=48)"/>

**Conclusion**: The RWKV baseline implementation is significantly better than FLA-RWKV in training speed. The speed of FLA-RWKV is about 83% of RWKV. With JIT enabled, RWKV's speed improves by about 11-12%.

## Summary

1. **Initialization has a significant impact**: FLA-RWKV performs poorly with default initialization, and even after using the initialization weights from the RWKV baseline implementation, it still performs significantly worse than RWKV.
2. **Training speed difference**: The RWKV baseline implementation is significantly better than FLA-RWKV in training speed. The speed of FLA-RWKV is about 83% of RWKV. With JIT enabled, RWKV's speed improves by about 11-12%.
