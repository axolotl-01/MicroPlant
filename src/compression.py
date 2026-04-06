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


def quantize_model(model, train_loader, val_loader, teacher=None,
                   epochs=8, lr=0.001, weight_decay=1e-4, qconfig='fbgemm',
                   save_name='quantized_model', device='DEVICE'):
    model.train()
    model.to('cpu')
    model.qconfig = quant.get_default_qat_qconfig(qconfig)
    qat_model = quant.prepare_qat(model, inplace=False)
    qat_model.to(device)

    qat_model = train_model(
        qat_model, train_loader, val_loader, epochs,
        teacher=teacher, l1_lambda=0.0, lr=lr, weight_decay=weight_decay,
        save_name=save_name, device=device
    )

    qat_model.eval()
    qat_model.to('cpu')
    quantized_model = quant.convert(qat_model, inplace=False)

    return quantized_model