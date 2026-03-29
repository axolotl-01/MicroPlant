import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_model_bytes(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    total_size_kb = (param_size + buffer_size) / 1024
    return total_size_kb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)