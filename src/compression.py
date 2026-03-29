import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=3.0):
        super().__init__()
        self.alpha = alpha
        self.temp = temp

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        student_soft = F.log_softmax(student_logits / self.temp, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temp, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return (1 - self.alpha) * hard_loss + self.alpha * (self.temp ** 2) * soft_loss

class FeatureDistillationLoss(nn.Module):
    def __init__(self, crit_weight=1.0):
        super().__init__()
        self.crit_weight = crit_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        loss = self.mse_loss(student_features, teacher_features)
        return loss * self.crit_weight

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