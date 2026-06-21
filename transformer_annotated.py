"""
Transformer 实现（带详细注释版）

架构参考 LLaMA，主要组件：
  - RMSNorm：比 LayerNorm 更快的归一化，省去均值计算
  - RoPE：旋转位置编码，支持相对位置，外推能力强
  - GQA：分组查询注意力，K/V head 数 < Q head 数，节省显存
  - SwiGLU FFN：门控前馈网络，比 ReLU FFN 效果更好
  - Pre-Norm + 残差：训练更稳定
  - Weight Tying：embedding 权重复用为 lm_head
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# 1. RMSNorm
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    公式：y = (x / RMS(x)) * γ
    RMS(x) = sqrt(mean(x²) + ε)

    相比 LayerNorm，去掉了均值计算（重中心化），只保留缩放（重缩放）。
    速度快约 7~15%，效果基本持平。
    """

    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        # γ：可学习的逐维度缩放参数，初始化为 1（训练开始时不改变分布）
        self.weight = nn.Parameter(torch.ones(dim))
        # ε：防止 RMS 为 0 时除以 0，Python float 不需要移动到 GPU
        self.eps = eps

    def forward(self, x):
        # x.pow(2)：每个元素平方，shape 不变
        # .mean(dim=-1, keepdim=True)：在特征维取均值，shape: (..., 1)
        # .add(self.eps)：加 ε 防止开根号时分母为 0
        # .sqrt()：得到 RMS，shape: (..., 1)，会广播到 x 的形状
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # (1 / rms) * x：归一化
        # * self.weight：逐维度乘以可学习参数 γ，恢复表达能力
        return x * (1 / rms) * self.weight


# ─────────────────────────────────────────────────────────────
# 2. RoPE（旋转位置编码）
# ─────────────────────────────────────────────────────────────

def pre_compute_rope(dim: int, max_len: int = 160000, base: int = 10000):
    """
    预计算 RoPE 所需的 cos/sin 表，在模型初始化时调用一次。

    每对维度 (2i, 2i+1) 对应一个旋转频率：
      θ_i = 1 / (base ^ (2i / dim))

    维度越小频率越高（变化越快），越大频率越低（变化越慢），
    类似傅里叶分解捕捉不同粒度的位置信息。

    返回:
      cos: shape (max_len, dim/2)
      sin: shape (max_len, dim/2)
    """
    # arange(0, dim, 2) → [0, 2, 4, ..., dim-2]，即 2i
    # / dim → 2i/d
    # pow(base, ...) → base^(2i/d)
    # 1 / ... → θ_i，shape: (dim/2,)
    freqs = 1 / torch.pow(base, torch.arange(0, dim, 2).float() / dim)

    # 位置序列 [0, 1, 2, ..., max_len-1]，shape: (max_len,)
    t = torch.arange(max_len).float()

    # outer product：freqs_table[pos][i] = pos × θ_i（旋转角度）
    # shape: (max_len, dim/2)
    freqs = torch.outer(t, freqs)

    cos = torch.cos(freqs)  # shape: (max_len, dim/2)
    sin = torch.sin(freqs)  # shape: (max_len, dim/2)
    return cos, sin


def apply_rope(x, cos, sin):
    """
    将旋转位置编码施加到 Q 或 K。

    旋转公式（对每对维度 (x1, x2)）：
      x1' = x1 * cos - x2 * sin
      x2' = x1 * sin + x2 * cos

    x shape: (B, n_heads, T, head_dim)
    cos/sin shape: (T, head_dim/2)，会自动广播
    """
    # 按奇偶拆分，对应旋转公式的 (x1, x2) 维度对
    x1 = x[..., ::2]   # 偶数维度，shape: (B, n_heads, T, head_dim/2)
    x2 = x[..., 1::2]  # 奇数维度，shape: (B, n_heads, T, head_dim/2)

    # 拼接旋转后的两部分，顺序与拆分时保持一致
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ─────────────────────────────────────────────────────────────
# 3. Attention（多头注意力 + GQA）
# ─────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    带 GQA 的多头注意力。

    GQA（Grouped Query Attention）：n_kv_heads 组 K/V 被 n_heads 组 Q 共享。
    每 (n_heads // n_kv_heads) 个 Q head 共享同一组 K/V。
    n_kv_heads == n_heads 时退化为标准 MHA。
    n_kv_heads == 1 时为 MQA（所有 Q 共享一组 K/V）。
    """

    def __init__(self, n_heads: int = 8, n_kv_heads: int = 8, dim: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = dim
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.head_dim = dim // n_heads  # 每个 head 的特征维度

        # Q 投影：输出所有 n_heads 个 head
        self.W_Q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # K/V 投影：只输出 n_kv_heads 个 head（GQA 节省参数和 KV cache）
        self.W_K = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # 输出投影：把所有 head 的输出合并回 dim
        self.W_O = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None):
        B, T, dim = x.shape

        # 线性投影 + reshape：(B, T, dim) → (B, n_heads, T, head_dim)
        # transpose(1, 2)：把 head 维提前，方便后续矩阵乘法
        q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 施加 RoPE：只旋转 Q 和 K，不旋转 V
        # cos[:T], sin[:T]：截取当前序列长度的位置编码
        q_rope = apply_rope(q, cos[:T], sin[:T])
        k_rope = apply_rope(k, cos[:T], sin[:T])

        # GQA 展开：把 K/V 复制 ratio 次，与 Q 的 head 数对齐
        # [k0, k1, k2, k3] → [k0, k0, k1, k1, k2, k2, k3, k3]（ratio=2 时）
        ratio = self.n_heads // self.n_kv_heads
        repeat_k = k_rope.repeat_interleave(ratio, dim=1)  # (B, n_heads, T, head_dim)
        repeat_v = v.repeat_interleave(ratio, dim=1)        # (B, n_heads, T, head_dim)

        # Attention score：Q @ K^T / sqrt(head_dim)
        # 除以 sqrt(head_dim)：防止点积过大导致 softmax 梯度消失
        # shape: (B, n_heads, T, T)
        attn_scores = q_rope @ repeat_k.transpose(-2, -1) / self.head_dim ** 0.5

        # 因果掩码：下三角矩阵，token i 只能看到位置 <= i 的 token
        # 未来位置填 -inf，softmax 后变为 0
        causal_mask = torch.tril(torch.ones(T, T, device=q_rope.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # softmax 归一化（在 key 维上），shape: (B, n_heads, T, T)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

        # 加权求和：shape: (B, n_heads, T, head_dim)
        out = attn_scores @ repeat_v

        # 合并所有 head：(B, n_heads, T, head_dim) → (B, T, n_heads * head_dim)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)

        # 输出投影，shape: (B, T, dim)
        return self.W_O(out)


# ─────────────────────────────────────────────────────────────
# 4. FFN（SwiGLU 前馈网络）
# ─────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """
    SwiGLU 前馈网络。

    公式：FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )

    三个投影：
      W_gate：计算门控信号，经 SiLU 后决定哪些特征通过
      W_up：计算信息信号
      W_down：降回原始维度

    两条路的乘积（GLU 门控）比单路 ReLU FFN 效果更好。
    注意：是 silu(gate) * up，不是 silu(gate * up)。
    """

    def __init__(self, dim: int = 512, dim_ff: int = 2048):
        super().__init__()
        self.W_GATE = nn.Linear(dim, dim_ff, bias=False)
        self.W_UP   = nn.Linear(dim, dim_ff, bias=False)
        self.W_DOWN = nn.Linear(dim_ff, dim, bias=False)

    def forward(self, x):
        # SiLU(W_gate(x))：门控，平滑地决定信息通过量
        # W_up(x)：信息
        # 逐元素相乘：门控筛选
        # W_down：降维回 dim
        return self.W_DOWN(torch.nn.functional.silu(self.W_GATE(x)) * self.W_UP(x))


# ─────────────────────────────────────────────────────────────
# 5. TransformerBlock
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    一个 Transformer 层。

    结构（Pre-Norm）：
      x → RMSNorm → Attention → + x  →  RMSNorm → FFN → + x
                                 ↑残差                     ↑残差

    Pre-Norm：在子层之前归一化，训练比 Post-Norm 更稳定。
    残差连接：梯度可以直接流回输入，避免深层网络梯度消失。
    """

    def __init__(self, n_heads: int = 8, n_kv_heads: int = 8,
                 dim: int = 512, dim_ff: int = 2048):
        super().__init__()
        self.rms_norm_attn = RMSNorm(dim)
        self.attention = Attention(n_heads, n_kv_heads, dim)
        self.rms_norm_ffn = RMSNorm(dim)
        self.ffn = FFN(dim, dim_ff)

    def forward(self, x, cos=None, sin=None):
        # Attention 子层：Pre-Norm + 计算 + 残差
        attn_norm = self.rms_norm_attn(x)
        attn = self.attention(attn_norm, cos=cos, sin=sin)
        attn_residual = x + attn          # 残差：保留原始信息

        # FFN 子层：Pre-Norm + 计算 + 残差
        ffn_norm = self.rms_norm_ffn(attn_residual)
        ffn = self.ffn(ffn_norm)
        ffn_residual = attn_residual + ffn  # 残差

        return ffn_residual


# ─────────────────────────────────────────────────────────────
# 6. Transformer（完整模型）
# ─────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    完整的 Decoder-only Transformer。

    数据流：
      input_ids (B, T)
        → Embedding → (B, T, dim)
        → N × TransformerBlock
        → RMSNorm
        → @ embedding.weight.T → logits (B, T, vocab_size)

    关键设计：
      - register_buffer：cos/sin 不参与训练，但需跟随模型迁移到 GPU
      - Weight Tying：lm_head 复用 embedding 权重，减少参数，共享语义空间
    """

    def __init__(self, vocab_size: int = 10000, block_num: int = 8,
                 n_heads: int = 8, n_kv_heads: int = 8,
                 dim: int = 512, dim_ff: int = 2048,
                 rope_max_len: int = 160000):
        super().__init__()

        # 预计算 RoPE cos/sin 表（只算一次，不在 forward 重复计算）
        cos, sin = pre_compute_rope(dim // n_heads, max_len=rope_max_len)

        # register_buffer：注册为 buffer 而非 Parameter
        #   - 随 model.to('cuda') 自动移到 GPU
        #   - 出现在 state_dict()，随 checkpoint 保存
        #   - 不出现在 parameters()，优化器不更新
        # 必须在普通属性赋值之前调用，否则 self.__dict__ 会遮蔽 self._buffers
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Token embedding 表：vocab_size 个 token，每个映射到 dim 维向量
        self.embedding = nn.Embedding(vocab_size, dim)

        # N 个 Transformer Block，用 ModuleList 注册
        # ModuleList 保证子模块能被 parameters()/to()/state_dict() 正确处理
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, n_kv_heads, dim, dim_ff)
            for _ in range(block_num)
        ])

        # 最后一层归一化（在 lm_head 之前）
        self.final_norm = RMSNorm(dim)

        # 不在这里定义 lm_head，直接在 forward 里用 embedding.weight.t()
        # 原因：self.lm_head = self.embedding.weight 会触发 PyTorch __setattr__
        # 把 Parameter 重复注册，导致 state_dict 和 parameters() 出现两份

    def forward(self, input_ids):
        # input_ids: (B, T)，dtype=torch.long
        x = self.embedding(input_ids)   # → (B, T, dim)

        # 逐层过 Transformer Block
        # cos/sin 传入供各层的 Attention 做 RoPE
        for block in self.blocks:
            x = block(x, self.cos, self.sin)

        # 最终归一化
        x = self.final_norm(x)          # → (B, T, dim)

        # lm_head（权重共享）：
        # embedding.weight shape: (vocab_size, dim)
        # .t() → (dim, vocab_size)
        # x @ .t() → (B, T, vocab_size)
        # 返回 logits，调用方自行 softmax 或直接用于 cross_entropy
        output = x @ self.embedding.weight.t()
        return output


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vocab_size = 10000
    model = Transformer(vocab_size, block_num=8, n_heads=8, dim=512)
    inputs = torch.randint(0, vocab_size, (2, 128), dtype=torch.long)
    output_logits = model(inputs)
    print(output_logits.shape)  # → torch.Size([2, 128, 10000])
