# Transformer（完整模型）

## 整体结构

```
input_ids (token 整数序列)
      │
      ▼
  Embedding       → token id 映射为向量，shape: (B, T, dim)
      │
      ▼
  register_buffer → cos/sin（RoPE 预计算表，随模型移动到 GPU）
      │
      ▼
  Block × N       → 堆叠 N 个 TransformerBlock
      │
      ▼
  RMSNorm         → 最终归一化
      │
      ▼
  x @ embedding.weight.T   → 投影回 vocab 空间，shape: (B, T, vocab_size)
      │
      ▼
  logits（未经 softmax，调用方自行处理）
```

## 代码解读

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size=10000, block_num=8, n_heads=8,
                 n_kv_heads=8, dim=512, dim_ff=2048, rope_max_len=160000):
        super().__init__()

        # RoPE 预计算：在 __init__ 阶段一次性算好，不在 forward 重复计算
        # 必须在 register_buffer 之前算好，再注册
        cos, sin = pre_compute_rope(dim // n_heads, max_len=rope_max_len)

        # register_buffer：cos/sin 是常量，不参与训练
        # 但需要随 model.to('cuda') 自动移到 GPU，并保存到 checkpoint
        # 必须在普通属性赋值之前调用，否则 __dict__ 会遮蔽 _buffers
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Token embedding 表：vocab_size × dim
        self.embedding = nn.Embedding(vocab_size, dim)

        # N 个 Transformer Block，用 ModuleList 注册（能被 parameters() 遍历到）
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, n_kv_heads, dim, dim_ff)
            for _ in range(block_num)
        ])

        # 最终归一化（输出头之前）
        self.final_norm = RMSNorm(dim)

    def forward(self, input_ids):
        # input_ids: (B, T)，整数 token id
        x = self.embedding(input_ids)   # → (B, T, dim)

        # 逐层过 Transformer Block，传入 cos/sin 供 RoPE 使用
        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        x = self.final_norm(x)          # → (B, T, dim)

        # lm_head：用 embedding 权重的转置投影到词表空间（权重共享）
        # embedding.weight shape: (vocab_size, dim)
        # .t() → (dim, vocab_size)
        # x @ .t() → (B, T, vocab_size)
        output = x @ self.embedding.weight.t()

        return output  # logits，调用方自行 softmax 或 cross_entropy
```

## 权重共享（Weight Tying）

`output = x @ self.embedding.weight.t()` 复用了 embedding 的权重矩阵，而不是单独定义一个 `lm_head` 线性层。

好处：
- 减少参数量（vocab_size × dim 这部分只存一份）
- 实践中效果更好：embedding 和 output 投影共享同一语义空间

注意：不能写 `self.lm_head = self.embedding.weight`，PyTorch 的 `__setattr__` 会把 Parameter 再注册一次，导致重复出现在 `state_dict()` 和 `parameters()` 里。应直接在 `forward` 里用 `self.embedding.weight.t()`。

## 参数量估算（dim=512, vocab=10000, 8层）

| 模块 | 参数量 |
|------|--------|
| Embedding | vocab × dim = 10000 × 512 = 5.12M |
| 每层 Attention (W_Q/K/V/O) | 4 × dim² = 4 × 512² = 1.05M |
| 每层 FFN (W_gate/up/down) | 3 × dim × dim_ff = 3 × 512 × 2048 = 3.15M |
| 每层 RMSNorm × 2 | 2 × dim = 1024（可忽略） |
| **8层合计** | 8 × (1.05 + 3.15)M = 33.6M |
| **总计** | ≈ 38.7M |
