# RWKV 与 FLA-RWKV 参数命名对齐

本文档说明了 RWKV 和 FLA 模型之间的参数命名对应关系。除命名外，如果存在形状不一致的情况会在表格的括号中注明。

## 1. LoRA-w

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| w1 ($[N_\text{embedding}, D_\text{decay-lora}]$) | w_lora.lora.0.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| w2 ($[N_\text{embedding}, D_\text{decay-lora}]$) | w_lora.2.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| w0 ($[1,1,N_\text{embedding}]$) | w_lora.2.bias ($[N_\text{embedding}]$) | 需要reshape |

**对应关系**：

- RWKV: `w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5`
- FLA: `self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')`

## 2. LoRA-v

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| v1 ($[N_\text{embedding}, D_\text{decay-lora}]$) | v_lora.0.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| v2 ($[N_\text{embedding}, D_\text{decay-lora}]$) | v_lora.2.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| v0 ($[1,1,N_\text{embedding}]$) | v_lora.2.bias ($[N_\text{embedding}]$) | 需要reshape |

**对应关系**：

- RWKV: `v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)`
- FLA: `self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)`

## 3. LoRA-a

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| a1 ($[N_\text{embedding}, D_\text{decay-lora}]$) | a_lora.0.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| a2 ($[N_\text{embedding}, D_\text{decay-lora}]$) | a_lora.2.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| a0 ($[1,1,N_\text{embedding}]$) | a_lora.2.bias ($[N_\text{embedding}]$) | 需要reshape |

**对应关系**：

- RWKV: `a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)`
- FLA: `self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)`

## 4. LoRA-g

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| g1 ($[N_\text{embedding}, D_\text{decay-lora}]$) | g_lora.lora.0.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |
| g2 ($[N_\text{embedding}, D_\text{decay-lora}]$) | g_lora.lora.2.weight $[D_\text{decay-lora},N_\text{embedding}]$ | 需要转置 |

**对应关系**：

- RWKV: `g = torch.sigmoid(xg @ self.g1) @ self.g2`
- FLA: `self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)`

## 5. k_k, k_a

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| k_k ($[1,1,N_\text{embedding}]$) | k_k ($[N_\text{embedding}]$) | 需要reshape |
| k_a ($[1,1,N_\text{embedding}]$) | k_a ($[N_\text{embedding}]$) | 需要reshape |

**对应关系**：

- RWKV: `nn.Parameter(torch.zeros(1,1,C)+...)`
- FLA: `nn.Parameter(torch.zeros(self.key_dim))`

## 6. 投影层 (r, k, v, o)

| RWKV 参数 | FLA 参数 |
|-----------|----------|
| receptance | r_proj |
| key | k_proj |
| value | v_proj |
| output | o_proj |

## 7. GroupNorm

| RWKV 参数 | FLA 参数 | 说明 |
|-----------|----------|------|
| ln_x | g_norm | |

**对应关系**：

- RWKV: `self.ln_x = nn.GroupNorm(H, C, eps=64e-5)`
- FLA: `self.g_norm = nn.GroupNorm(num_groups=self.num_heads, ...)`

## 权重转换

在 `train-temp/train.py` 中提供了 `StateTransf_rwkv2fla` 函数，用于将 RWKV 的权重转换为 FLA 格式。该函数处理了以下转换：

1. **直接映射**：参数名称和形状完全相同
2. **Reshape**：从 `[1,1,N]` 转换为 `[N]`
3. **转置**：权重矩阵需要转置以匹配 FLA 的实现

转换逻辑详见 [train.py](./train-temp/train.py) 中的实现。
