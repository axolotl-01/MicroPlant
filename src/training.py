import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from src.compression import DistillationLoss, FeatureDistillationLoss

def train_model(model, train_loader, val_loader, epochs, lr, teacher, device, save_name='microplant'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    if teacher:
        criterion = DistillationLoss(alpha=0.5, temp=3.0) 
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    model.to(device)
    if teacher: 
        teacher.to(device)
        teacher.eval()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(X)
            if teacher:
                with torch.inference_mode():
                    t_out = teacher(X)
                loss = criterion(out, t_out, y)
            else:
                loss = criterion(out, y)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                v_loss = nn.CrossEntropyLoss()(out, y)
                val_loss += v_loss.item()
                _, pred = out.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            if not os.path.exists('models'): os.makedirs('models')
            torch.save(model.state_dict(), f'models/{save_name}_best.pth')

        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    return model