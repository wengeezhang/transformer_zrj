# RoPE（Rotary Position Embedding）

## 核心思想

不直接把位置编码**加**到 token embedding 上，而是在计算 attention score 时，把位置信息**旋转进** Q 和 K。

好处：天然支持相对位置，外推能力强（LLaMA、Qwen 等都用它）。

## 频率计算

每两个维度分配一个旋转频率：

$$
\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, \frac{d}{2}-1
$$

维度越靠前，频率越高（变化越快）；维度越靠后，频率越低（变化越慢）。类似傅里叶分解，用不同频率捕捉不同粒度的位置信息。

## 旋转公式

对于位置 t，维度对 (x₁, x₂)：

$$
\begin{bmatrix} x_1' \\ x_2' \end{bmatrix} =
\begin{bmatrix} \cos(t\theta) & -\sin(t\theta) \\ \sin(t\theta) & \cos(t\theta) \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

即：
$$
x_1' = x_1 \cos(t\theta) - x_2 \sin(t\theta)
$$
$$
x_2' = x_1 \sin(t\theta) + x_2 \cos(t\theta)
$$

## 代码解读

### pre_compute_rope：预计算 cos/sin 表

```python
def pre_compute_rope(dim: int, max_len: int = 160000, base: int = 10000):
    # 计算每个维度对的频率，shape: (dim/2,)
    # arange(0, dim, 2) → [0, 2, 4, ..., dim-2]，即 2i
    # / dim             → 2i/d
    # pow(base, ...)    → base^(2i/d)
    # 1 /               → θ_i = 1 / base^(2i/d)
    freqs = 1 / torch.pow(base, torch.arange(0, dim, 2).float() / dim)

    # 位置序列，shape: (max_len,)
    t = torch.arange(max_len).float()

    # 外积：每个位置 × 每个频率，shape: (max_len, dim/2)
    # freqs[i][j] = position_i × θ_j，即旋转角度
    freqs = torch.outer(t, freqs)

    cos = torch.cos(freqs)  # shape: (max_len, dim/2)
    sin = torch.sin(freqs)  # shape: (max_len, dim/2)
    return cos, sin
```

### apply_rope：将旋转编码应用到 Q/K

```python
def apply_rope(x, cos, sin):
    # x shape: (B, n_heads, T, head_dim)
    # 按奇偶拆分，对应旋转公式的 (x1, x2) 维度对
    x1 = x[..., ::2]   # 偶数维度，shape: (B, n_heads, T, head_dim/2)
    x2 = x[..., 1::2]  # 奇数维度，shape: (B, n_heads, T, head_dim/2)

    # cos/sin shape 从 (T, dim/2) 广播到 (B, n_heads, T, head_dim/2)
    # 旋转公式：
    # x1' = x1*cos - x2*sin
    # x2' = x1*sin + x2*cos
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

## 为什么用 register_buffer 存 cos/sin

cos/sin 表是**常量**，不参与训练（不需要梯度），但需要：
1. 随 `model.to('cuda')` 自动移到 GPU
2. 随 `state_dict()` 保存到 checkpoint

因此用 `register_buffer` 而不是普通属性赋值。
