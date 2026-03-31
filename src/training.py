import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=3.0):
        super().__init__()
        self.alpha = alpha
        self.temp = temp

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        student_soft = F.log_softmax(student_logits / self.temp, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temp, dim=1)
        soft_loss = torch.sum(teacher_soft * (teacher_soft.log() - student_soft)) / student_soft.size()[0] * (self.temp**2)
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss
    

def train_one_epoch(model, loader, optimizer, criterion, teacher=None, l1_lambda=0.0, device=DEVICE):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for X, y in tqdm(loader, desc='Training'):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        if teacher is not None:
            with torch.inference_mode():
                t_out = teacher(X)
            loss = criterion(out, t_out, y)
        else:
            loss = criterion(out, y)

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), f1


def validate(model, loader, device=DEVICE):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), f1


def train_model(model, train_loader, val_loader, epochs, teacher=None, kd_alpha=0.5, kd_temp=3.0,
                l1_lambda=0.0, lr=0.001, weight_decay=1e-4, save_name='model', device=DEVICE):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    if teacher is not None:
        criterion = KnowledgeDistillationLoss(alpha=kd_alpha, temp=kd_temp)
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion,
                                                teacher=teacher, l1_lambda=l1_lambda, device=DEVICE)
        val_loss, val_f1 = validate(model, val_loader, device=device)
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'{save_name}_best.pth')

        print(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_f1:.2f} | "
              f"Val Loss {val_loss:.4f} F1 {val_f1:.4f}")

    model.load_state_dict(torch.load(f'{save_name}_best.pth'))
    return model