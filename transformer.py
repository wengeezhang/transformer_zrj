import torch
import torch.nn as nn

vocab_size = 10000
input_dim = 512

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

class Attention(nn.Module):
    def __init__(self, n_heads: int = 8, dim: int = 512):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.head_dim = dim // n_heads

        # linear layers for each head
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return x

class FFN(nn.Module):
    def __init__(self):
        super().__init__()



class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int = 8, dim: int = 512):
        self.rms_norm_attn = RMSNorm(input_dim)
        self.attention = Attention(n_heads, dim)
        self.rms_norm_ffn = RMSNorm(input_dim)
        self.ffn = FFN()
        super().__init__()
    
    def forward(self, x):
        # pre-normalization for attention
        attn_norm = self.rms_norm_attn(x)
        # attention
        attn = self.attention(attn_norm)
        # residual connection for attention
        attn_residual = x + attn
        # pre-normalization for FFN
        ffn_norm = self.rms_norm_ffn(attn_residual)
        ffn = self.ffn(ffn_norm)
        ffn_residual = attn_residual + ffn
        return ffn_residual

class Transformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, block_num: int = 8, n_heads: int = 8, dim: int = 512):
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.blocks = nn.ModuleList([TransformerBlock(n_heads, dim) for _ in range(block_num)])
        self.final_norm = nn.LayerNorm(input_dim)
        # lm_head use embedding weight
        self.lm_head = self.embedding.weight
        super().__init__()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        output = self.lm_head(x)
        # caller will apply softmax later
        return output
        
model = Transformer(vocab_size, block_num=8, n_heads=8, dim=512)
inputs = torch.randint(0, vocab_size, (2, 128), dtype=torch.long)
output_logits = model(inputs)
