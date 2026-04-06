import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
from torch.nn.utils import prune    
from src.training import train_model


def apply_global_pruning(model, sparsity=0.5):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    print(f"Applied global pruning with sparsity {sparsity*100:.0f}%")


def remove_pruning_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except:
                pass


def apply_dynamic_quantization(model, original_model_path, save_name="../models/quantized"):

    model.to('cpu')
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    full_save_path = f"{save_name}.pth"
    torch.save(quantized_model.state_dict(), full_save_path)
    
    original_size = os.path.getsize(original_model_path) / 1024
    quant_size = os.path.getsize(full_save_path) / 1024
    
    print(f"Saved to: {full_save_path}")
    print(f"Original Size: {original_size:.2f} KB")
    print(f"Quantized Size: {quant_size:.2f} KB")
    print(f"Reduction: {original_size / quant_size:.1f}x smaller")
    
    return quantized_model


def quantize_model(model, train_loader, val_loader, teacher=None,
                   epochs=8, lr=0.001, weight_decay=1e-4, qconfig='fbgemm',
                   save_name='quantized_model', device='cpu'):
    model.train()
    model.to(device)
    model.qconfig = quant.get_default_qat_qconfig(qconfig)
    qat_model = quant.prepare_qat(model, inplace=False)

    qat_model = train_model(
        qat_model, train_loader, val_loader, epochs,
        teacher=teacher, l1_lambda=0.0, lr=lr, weight_decay=weight_decay,
        save_name=save_name, device=device
    )

    qat_model.eval()
    quantized_model = quant.convert(qat_model, inplace=False)

    return quantized_model