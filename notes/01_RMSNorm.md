# RMSNorm

## 公式

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

## 与 LayerNorm 的区别

| | LayerNorm | RMSNorm |
|--|-----------|---------|
| 计算均值 | ✅ | ❌ 省掉 |
| 计算方差 | ✅ | 用 RMS 代替 |
| 参数 | γ + β | 只有 γ |
| 速度 | 基准 | 快约 7~15% |

RMSNorm 的假设：**重缩放比重中心化更重要**。去掉均值计算对效果影响极小。

## 代码解读

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # γ，可学习缩放参数，初始化为 1
        self.eps = eps                               # 防止除以 0

    def forward(self, x):
        # x.pow(2)               → 每个元素平方
        # .mean(dim=-1)          → 在最后一维（特征维）取均值，得到 RMS² per token
        # keepdim=True           → 保持维度，方便后续广播
        # .add(self.eps)         → 加 eps 防止开根号时分母为 0
        # .sqrt()                → 得到 RMS，shape: (B, T, 1)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # x / rms                → 归一化
        # * self.weight          → 乘以可学习的 γ，恢复表达能力
        return x * (1 / rms) * self.weight
```

## γ 参数的作用

纯归一化后所有维度量级相同，模型无法区分哪些维度更重要。γ 是逐维度的缩放系数，让模型在训练中自己学习每个维度的重要性。初始化为 1 表示训练开始时不改变分布。

## 使用位置

Pre-Norm 结构：在 Attention 和 FFN **之前**归一化，而不是之后。

```
x → RMSNorm → Attention → + x（残差）→ RMSNorm → FFN → + x（残差）
```

Pre-Norm 相比 Post-Norm 训练更稳定，梯度不容易消失。
