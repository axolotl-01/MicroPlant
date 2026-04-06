import torch
import torch.nn as nn
from torchvision import models

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.use_se = use_se
        if use_se:
            reduced = max(1, in_ch // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, reduced, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced, in_ch, 1),
                nn.Sigmoid()
            )
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = self.bn1(x)
        if self.use_se:
            x = x * self.se(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        
        if res.shape == x.shape:
            x = x + res
        x = self.act(x)
        return x

class MicroPlant(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(8, 16, stride=2, use_se=True),
            DepthwiseSeparableConv(16, 24, stride=2, use_se=True),
            DepthwiseSeparableConv(24, 32, stride=2, use_se=True),
            DepthwiseSeparableConv(32, 48, stride=1, use_se=True),
            DepthwiseSeparableConv(48, 64, stride=2, use_se=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_microplant(num_classes=4):
    return MicroPlant(num_classes=num_classes)

def get_teacher_model(num_classes=4):
    model = model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model