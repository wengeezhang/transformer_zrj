import torch
from transformer import Transformer

# test play. must be run on cuda
if __name__ == "__main__":
    # must be run on cuda
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available")
    
    vocab_size = 10000
    dim = 8
    model = Transformer(vocab_size=vocab_size, block_num=1, n_heads=1, n_kv_heads=1, dim=dim, dim_ff=16)
    inputs = torch.randint(0, vocab_size, (2,dim), dtype=torch.long)
    model.to("cuda")
    inputs = inputs.to("cuda")
    output_logits = model(inputs)
    print(output_logits.shape)