"""Models for context margin study."""
import torch
import torch.nn as nn
from torchvision import models

class SingleStreamResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class DualStreamFusion(nn.Module):
    def __init__(self, num_classes=6, dropout_p=0.3, ablation_mode="fusion_dropout"):
        """
        ablation_mode: "tight_only", "context_only", "fusion_nodrop", "fusion_dropout"
        """
        super().__init__()
        # Shared backbone
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1]) # up to pooling
        in_features = base.fc.in_features
        
        self.ablation_mode = ablation_mode
        self.dropout_p = dropout_p
        
        if "fusion" in ablation_mode:
            self.fc = nn.Linear(in_features * 2, num_classes)
        else:
            self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x_tight, x_context):
        if self.ablation_mode == "tight_only":
            f = self.features(x_tight).view(x_tight.size(0), -1)
            return self.fc(f)
            
        if self.ablation_mode == "context_only":
            f = self.features(x_context).view(x_context.size(0), -1)
            return self.fc(f)
            
        # Fusion modes
        f_tight = self.features(x_tight).view(x_tight.size(0), -1)
        f_context = self.features(x_context).view(x_context.size(0), -1)
        
        # Context Dropout
        if self.ablation_mode == "fusion_dropout" and self.training:
            # Mask out context features with probability p
            mask = (torch.rand(f_context.size(0), 1, device=f_context.device) > self.dropout_p).float()
            f_context = f_context * mask
            
        f_fused = torch.cat([f_tight, f_context], dim=1)
        return self.fc(f_fused)

class SingleStreamConvNeXt(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier[2] = nn.Linear(self.backbone.classifier[2].in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
