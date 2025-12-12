"""
EfficientNet-B2 model for ASVspoof detection
Uses pre-trained EfficientNet-B2 backbone with custom classifier head
Optimized for mel-spectrogram input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B2_Weights


class Model(nn.Module):
    """
    EfficientNet-B2 for mel spectrogram input (--feature 1).
    Input: (batch_size, n_mels, frames) - typically (B, 128, T)
    
    EfficientNet-B2 specs:
    - Input: 260x260x3
    - Parameters: ~9.2M
    - Top-1 Accuracy: 80.5%
    - Compound scaling: depth=1.1, width=1.1, resolution=260
    """
    def __init__(self, d_args):
        super().__init__()
        
        dropout = d_args.get("dropout", 0.3)
        pretrained = d_args.get("pretrained", True)
        freeze_backbone = d_args.get("freeze_backbone", False)
        
        # Load pre-trained EfficientNet-B2
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)
        
        # Modify first conv layer to accept single channel input (mel-spectrogram)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,  # Single channel input
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )
        
        # Initialize new first conv layer
        if pretrained:
            # Average the pretrained RGB weights to single channel
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_first_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Freeze backbone if requested (for transfer learning)
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get the number of features from EfficientNet-B2
        # EfficientNet-B2 outputs 1408 features
        num_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head for spoofing detection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, Freq_aug=False):
        """
        Forward pass
        Args:
            x: input tensor of shape (B, n_mels, frames) or (B, 1, n_mels, frames)
            Freq_aug: frequency augmentation flag (not used here)
        Returns:
            embeddings: feature embeddings
            output: logits for classification (B, 2)
        """
        # Ensure 4D input: (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, n_mels, frames) -> (B, 1, n_mels, frames)
        
        # Extract features using EfficientNet backbone
        features = self.backbone(x)  # (B, 1408)
        
        # Store embeddings
        embeddings = features.clone()
        
        # Classification
        output = self.classifier(features)  # (B, 2)
        
        return embeddings, output


class ModelWithAttention(nn.Module):
    """
    EfficientNet-B2 with attention pooling for better temporal modeling.
    Useful when input has variable length or needs better time-axis attention.
    """
    def __init__(self, d_args):
        super().__init__()
        
        dropout = d_args.get("dropout", 0.3)
        pretrained = d_args.get("pretrained", True)
        freeze_backbone = d_args.get("freeze_backbone", False)
        att_bottleneck = d_args.get("att_bottleneck", 128)
        
        # Load pre-trained EfficientNet-B2
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)
        
        # Modify first conv layer to accept single channel input
        original_first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )
        
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_first_conv.weight.mean(dim=1, keepdim=True)
                )
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Remove avgpool and classifier
        self.features_only = nn.Sequential(*list(self.backbone.features))
        
        # Feature dimension after last conv block (before pooling)
        # For EfficientNet-B2: 1408 channels
        self.feature_channels = 1408
        
        # Attention pooling over spatial dimensions
        self.attention = nn.Sequential(
            nn.Conv2d(self.feature_channels, att_bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(att_bottleneck, 1, kernel_size=1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels * 2, 512),  # *2 for mean + std
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, Freq_aug=False):
        """
        Forward pass with attention pooling
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract features: (B, C, H, W)
        features = self.features_only(x)
        
        # Attention pooling
        att_weights = self.attention(features)  # (B, 1, H, W)
        att_weights = F.softmax(att_weights.view(features.size(0), -1), dim=1)  # (B, H*W)
        att_weights = att_weights.view(features.size(0), 1, features.size(2), features.size(3))  # (B, 1, H, W)
        
        # Weighted statistics
        features_flat = features.view(features.size(0), features.size(1), -1)  # (B, C, H*W)
        att_weights_flat = att_weights.view(features.size(0), 1, -1)  # (B, 1, H*W)
        
        # Weighted mean and std
        mu = (features_flat * att_weights_flat).sum(dim=2)  # (B, C)
        sigma = torch.sqrt(
            torch.clamp((features_flat ** 2 * att_weights_flat).sum(dim=2) - mu ** 2, min=1e-12)
        )  # (B, C)
        
        embeddings = torch.cat([mu, sigma], dim=1)  # (B, C*2)
        
        # Classification
        output = self.classifier(embeddings)
        
        return embeddings, output
