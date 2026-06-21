# TransformerBlock

## 结构

```
x
│
├─► RMSNorm ─► Attention ─► + ──► x'
│                             │
└─────────────────────────────┘（残差连接）
│
├─► RMSNorm ─► FFN ──────► + ──► output
│                             │
└─────────────────────────────┘（残差连接）
```

两个子层，每个子层都是：**Pre-Norm → 子层计算 → 残差连接**。

## 代码解读

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_heads=8, n_kv_heads=8, dim=512, dim_ff=2048):
        super().__init__()
        self.rms_norm_attn = RMSNorm(dim)   # Attention 前的归一化
        self.attention = Attention(n_heads, n_kv_heads, dim)
        self.rms_norm_ffn = RMSNorm(dim)    # FFN 前的归一化
        self.ffn = FFN(dim, dim_ff)

    def forward(self, x, cos=None, sin=None):
        # --- Attention 子层 ---
        attn_norm = self.rms_norm_attn(x)           # Pre-Norm
        attn = self.attention(attn_norm, cos, sin)   # 计算注意力
        attn_residual = x + attn                     # 残差连接

        # --- FFN 子层 ---
        ffn_norm = self.rms_norm_ffn(attn_residual)  # Pre-Norm
        ffn = self.ffn(ffn_norm)                     # 前馈网络
        ffn_residual = attn_residual + ffn           # 残差连接

        return ffn_residual
```

## Pre-Norm vs Post-Norm

| | Post-Norm（原始 Transformer） | Pre-Norm（当前实现）|
|--|-------------------------------|---------------------|
| 位置 | 子层之后归一化 | 子层之前归一化 |
| 训练稳定性 | 较难，需要 warmup | 稳定，梯度不易消失 |
| 效果 | 收敛后略好 | 更容易训练深层网络 |

现代 LLM（LLaMA、GPT-4…）全部用 Pre-Norm。

## 残差连接的作用

残差连接让梯度有一条直接从输出流回输入的"高速公路"，避免深层网络的梯度消失。

数学上，若子层为 F(x)，则：
```
output = x + F(x)
∂output/∂x = 1 + ∂F/∂x
```

梯度至少为 1，不会因为层数增加而趋近于 0。
