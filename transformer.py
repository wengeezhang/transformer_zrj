import torch
import torch.nn as nn

vocab_size = 10000

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x * (1 / rms) * self.weight # element-wise division and multiplication

def pre_compute_rope(dim: int, max_len: int = 160000, base: int = 10000):
    # pre-compute the rope
    # unit_angle = 2 * torch.pi / max_len

    # prepare the angles for every position
    # angles = torch.arange(max_len) * unit_angle
    freqs = 1 / torch.pow(base, torch.arange(0, dim, 2).float() / dim)
    t = torch.arange(max_len).float()

    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(x, cos, sin):
    # use interleave split(not use block-wise split x.chunk(2, dim=-1))
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, n_heads: int = 8, n_kv_heads: int = 8, dim: int = 512):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = dim
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        self.head_dim = dim // n_heads

        # linear layers for each head
        self.W_Q = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos=None, sin=None):
        B, T, dim = x.shape
        q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # share k/v for every self.n_heads // self.n_kv_heads heads 
        # if self.n_heads // self.n_kv_heads == 1, every feature group has it's own k/v
        # if self.n_heads // self.n_kv_heads == self.n_heads, all feature groups share the single k/v
        k = self.W_K(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # rope
        q_rope = apply_rope(q, cos[:T], sin[:T])
        k_rope = apply_rope(k, cos[:T], sin[:T])
        # use repeat_interleave to copy k/v for feature groups that share the same k/v.
        # for example:
        # kv_heads is [head1, head2, head3, head4]
        # heads is                [head1, head2, head3, head4, head5, head6, head7, head8]
        # so repeat_interleave -> [head1, head1, head2, head2, head3, head3, head4, head4]
        repeat_k = k_rope.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        repeat_v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # calculate attention scores
        attn_scores_of_head = q_rope @ repeat_k.transpose(-2, -1) / torch.pow(self.head_dim, 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=q_rope.device))
        attn_scores_of_head = attn_scores_of_head.masked_fill(causal_mask == 0, float('-inf'))
        attn_scores_of_head = torch.nn.functional.softmax(attn_scores_of_head, dim=-1)
        attn_of_head = attn_scores_of_head @ repeat_v # output shape is (B, n_heads, T, head_dim)
        # transpose to (B, T, n_heads, head_dim)
        attn_of_head = attn_of_head.transpose(1, 2)

        # concat all heads: (B, T, n_heads, head_dim) -> (B, T, dim)
        attn_of_head = attn_of_head.reshape(B, T, self.n_heads * self.head_dim)
        # output projection
        return self.W_O(attn_of_head)

class FFN(nn.Module):
    def __init__(self, dim: int = 512, dim_ff: int = 2048):
        super().__init__()
        self.W_GATE = nn.Linear(dim, dim_ff, bias=False)
        self.W_UP = nn.Linear(dim, dim_ff, bias=False)
        self.W_DOWN = nn.Linear(dim_ff, dim, bias=False)
    def forward(self, x):
        return self.W_DOWN(torch.nn.functional.silu(self.W_GATE(x) * self.W_UP(x)))


class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int = 8, n_kv_heads: int = 8, dim: int = 512, dim_ff: int = 2048):
        super().__init__()
        self.rms_norm_attn = RMSNorm(dim, eps=1e-5)
        self.attention = Attention(n_heads, n_kv_heads, dim)
        self.rms_norm_ffn = RMSNorm(dim, eps=1e-5)
        self.ffn = FFN(dim, dim_ff)
    
    def forward(self, x, cos=None, sin=None):
        # pre-normalization for attention
        attn_norm = self.rms_norm_attn(x)
        # attention
        attn = self.attention(attn_norm, cos=cos, sin=sin)
        # residual connection for attention
        attn_residual = x + attn
        # pre-normalization for FFN
        ffn_norm = self.rms_norm_ffn(attn_residual)
        ffn = self.ffn(ffn_norm)
        ffn_residual = attn_residual + ffn
        return ffn_residual

class Transformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, block_num: int = 8, n_heads: int = 8, n_kv_heads: int = 8, dim: int = 512, dim_ff: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        cos, sin = pre_compute_rope(dim // n_heads, max_len=160000)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.blocks = nn.ModuleList([TransformerBlock(n_heads, n_kv_heads, dim, dim_ff) for _ in range(block_num)])
        self.final_norm = RMSNorm(dim)
        # lm_head use embedding weight
        self.lm_head = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        output = x @ self.lm_head.t()
        # caller will apply softmax later
        return output
        
model = Transformer(vocab_size, block_num=8, n_heads=8, dim=512)
inputs = torch.randint(0, vocab_size, (2, 128), dtype=torch.long)
output_logits = model(inputs)
