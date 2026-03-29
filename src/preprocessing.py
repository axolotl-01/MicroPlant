import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class make_dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = sorted(os.listdir(self.root))
        self.img_path = []
        self.label = []
        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(self.root, cls)
            if os.path.isdir(cls_path):
                for img in os.listdir(cls_path):
                    self.img_path.append(os.path.join(cls_path, img))
                    self.label.append(idx)
        
    def __len__(self):
        return len(self.label)

    def labels(self):
        return self.label
        
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        label = self.label[idx]
        return img, label

class apply_transform(Dataset):
    
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(base_root, batch_size=64, seed=0):

    set_seed(seed)
    
    aug_tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = make_dataset(base_root)
    labels = full_dataset.labels()

    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.1, random_state=seed, stratify=labels
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1, random_state=seed, stratify=train_val_labels
    )

    train_dataset = apply_transform(Subset(full_dataset, train_idx), transform=aug_tfs)
    val_dataset = apply_transform(Subset(full_dataset, val_idx), transform=tfs)
    test_dataset = apply_transform(Subset(full_dataset, test_idx), transform=tfs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.classes