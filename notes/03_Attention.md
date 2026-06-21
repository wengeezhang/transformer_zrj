# Attention（多头注意力 + GQA）

## 标准 Multi-Head Attention 回顾

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right) V
$$

每个 head 独立计算，最后拼接输出。

## GQA（Grouped Query Attention）

标准 MHA：每个 Query head 对应自己的 K/V head，共 n_heads 组 K/V。

GQA：多个 Query head **共享**一组 K/V，K/V head 数量 = n_kv_heads < n_heads。

```
n_heads=8, n_kv_heads=4：

Q heads:  [q0, q1, q2, q3, q4, q5, q6, q7]
K heads:  [k0,      k1,      k2,      k3  ]   ← 每个 K 被 2 个 Q 共享
```

好处：K/V cache 减小 n_heads/n_kv_heads 倍，推理时显存占用大幅降低。

特殊情况：
- `n_kv_heads == n_heads` → 标准 MHA
- `n_kv_heads == 1` → MQA（Multi-Query Attention，所有 Q 共享同一组 K/V）

## 代码解读

```python
class Attention(nn.Module):
    def __init__(self, n_heads=8, n_kv_heads=8, dim=512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads  # 每个 head 的维度

        # Q 投影：输出所有 query heads
        self.W_Q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # K/V 投影：只输出 n_kv_heads 组，GQA 的关键
        self.W_K = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # 输出投影：把所有 head 的输出合并回 dim
        self.W_O = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None):
        B, T, dim = x.shape

        # 线性投影后 reshape：(B, T, dim) → (B, n_heads, T, head_dim)
        q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 施加 RoPE：只对 Q 和 K 旋转，V 不旋转
        # cos[:T], sin[:T]：只取当前序列长度的位置编码
        q_rope = apply_rope(q, cos[:T], sin[:T])
        k_rope = apply_rope(k, cos[:T], sin[:T])

        # GQA：用 repeat_interleave 把 K/V 复制展开，与 Q 的数量对齐
        # 比如 n_heads=8, n_kv_heads=4：每个 K/V 复制 2 次
        # [k0, k1, k2, k3] → [k0, k0, k1, k1, k2, k2, k3, k3]
        ratio = self.n_heads // self.n_kv_heads
        repeat_k = k_rope.repeat_interleave(ratio, dim=1)
        repeat_v = v.repeat_interleave(ratio, dim=1)

        # Attention score：(B, n_heads, T, T)
        # 除以 sqrt(head_dim) 防止点积过大导致 softmax 梯度消失
        attn_scores = q_rope @ repeat_k.transpose(-2, -1) / self.head_dim ** 0.5

        # 因果掩码：下三角为 1，上三角填 -inf
        # token i 只能看到位置 <= i 的 token（自回归）
        causal_mask = torch.tril(torch.ones(T, T, device=q_rope.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # softmax：-inf → 0，归一化权重
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

        # 加权求和：(B, n_heads, T, T) @ (B, n_heads, T, head_dim) → (B, n_heads, T, head_dim)
        out = attn_scores @ repeat_v

        # 合并所有 head：(B, n_heads, T, head_dim) → (B, T, dim)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)

        # 输出投影
        return self.W_O(out)
```

## 为什么除以 sqrt(head_dim)

Q 和 K 的每个维度是独立随机变量，点积 `Q·K` 的方差 ≈ head_dim。

不缩放的话，head_dim 越大点积越大，softmax 的梯度趋近于 0（梯度消失）。除以 `sqrt(head_dim)` 把方差归一化为 1。

## 因果掩码

```
T=4 时的掩码（1=可见，0=不可见）：
  pos0  pos1  pos2  pos3
  [1,    0,    0,    0]   ← token 0 只看自己
  [1,    1,    0,    0]   ← token 1 看 0,1
  [1,    1,    1,    0]   ← token 2 看 0,1,2
  [1,    1,    1,    1]   ← token 3 看所有
```

`masked_fill(causal_mask == 0, -inf)` → 未来位置的 attention score 变为 -inf → softmax 后为 0。
