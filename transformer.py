import torch
import torch.nn as nn

vocab_size = 10000



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        return self.transformer(x)
        
model = Transformer(vocab_size)
inputs = torch.randint(0, vocab_size, (2, 128), dtype=torch.long)
output_logits = model(inputs)
