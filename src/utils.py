import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_model_bytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_bytes = param_size + buffer_size
    total_size_kb = total_size_bytes / 1024
    
    print(f"Model Weights: {param_size:,} bytes")
    print(f"Model Buffers: {buffer_size:,} bytes")
    print(f"Total Size: {total_size_kb:.2f} KB")
    return total_size_bytes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)