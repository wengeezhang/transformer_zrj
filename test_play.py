import torch
from transformer import Transformer

# test play. must be run on cuda
if __name__ == "__main__":
    # must be run on cuda
    # if not torch.cuda.is_available():
    #     raise ValueError("cuda is not available")
    
    vocab_size = 100
    dim = 8
    model = Transformer(vocab_size=vocab_size, block_num=1, n_heads=1, n_kv_heads=1, dim=dim, dim_ff=16)
    inputs = torch.randint(0, vocab_size, (2,dim), dtype=torch.long)
    #print(f"model.state dict: {model.state_dict()}")

    # print keys of model.state_dict()
    print(f"keys of model.state_dict(): {model.state_dict().keys()}")
    # print typeof of model.state_dict()["embedding.weight"]
    print(f"type of model.state_dict()['embedding.weight']: {type(model.state_dict()['embedding.weight'])}")
    # jsut print state dict keys of model.blocks.0
    print(f"keys of model.blocks.0.state dict(): {model.blocks[0].state_dict().keys()}")



    # model.to("cuda")
    # inputs = inputs.to("cuda")
    output_logits = model(inputs)
    print(output_logits.shape)