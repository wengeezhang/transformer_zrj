# FFN（SwiGLU 前馈网络）

## 经典 FFN（ReLU）

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

两层线性 + 一个激活函数，先升维再降维。

## SwiGLU（当前实现）

$$
\text{FFN}(x) = \text{SiLU}(x W_{gate}) \odot (x W_{up}) \cdot W_{down}
$$

三个投影矩阵，用**门控机制**控制信息流。

### 核心差异：GLU 门控

```
经典 FFN：  ReLU(x W1) × W2        ← 一条路
SwiGLU：   SiLU(x W_gate) × (x W_up) × W_down   ← 两条路相乘
```

`x W_gate` 经过 SiLU 后作为"门"，控制 `x W_up` 的哪些信息通过。两条路都是当前输入 x 的函数，门是动态的（依赖输入）。

### SiLU 激活函数

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

比 ReLU 平滑，负数区域不完全截断为 0，梯度更稳定。

## 代码解读

```python
class FFN(nn.Module):
    def __init__(self, dim=512, dim_ff=2048):
        super().__init__()
        # 三个投影，都无 bias（现代 LLM 惯例）
        self.W_GATE = nn.Linear(dim, dim_ff, bias=False)   # 门控路
        self.W_UP   = nn.Linear(dim, dim_ff, bias=False)   # 信息路
        self.W_DOWN = nn.Linear(dim_ff, dim, bias=False)   # 降维

    def forward(self, x):
        # silu(W_GATE(x))：门控信号，决定哪些特征"通过"
        # W_UP(x)：信息信号
        # 两者逐元素相乘：门控筛选信息
        # W_DOWN：降回原始维度
        return self.W_DOWN(torch.nn.functional.silu(self.W_GATE(x)) * self.W_UP(x))
```

## 常见错误：silu(gate * up) vs silu(gate) * up

```python
# ❌ 错误写法（不是 SwiGLU）
self.W_DOWN(torch.nn.functional.silu(self.W_GATE(x) * self.W_UP(x)))

# ✅ 正确写法（SwiGLU）
self.W_DOWN(torch.nn.functional.silu(self.W_GATE(x)) * self.W_UP(x))
```

SwiGLU 的定义是先对 gate 单独做 SiLU，再与 up 相乘。不是对两者的乘积做 SiLU。

## dim_ff 的选取

经典 FFN 通常 dim_ff = 4 × dim。

SwiGLU 因为有两个升维矩阵（W_gate 和 W_up），参数量多了约 50%。LLaMA 为了保持总参数量不变，把 dim_ff 缩小为约 2/3 × 4 × dim ≈ 8/3 × dim，实际取最近的 256 的整数倍。
