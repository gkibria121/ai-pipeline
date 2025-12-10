"""
LCNN (Light CNN) model for ASVspoof / Deepfake Audio Detection
Uses Max-Feature-Map (MFM) activation for robust feature learning.

This architecture is based on the LCNN design used in multiple top-performing
systems in ASVspoof challenges. MFM activation provides competitive feature
suppression which helps learn more discriminative features for spoofing detection.

Reference:
- Wu et al., "Light CNN for Deep Face Recognition with Noisy Labels", 2018
- Lavrentyeva et al., "Audio replay attack detection with deep learning frameworks", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map activation for 2D feature maps.
    Splits feature channels in half and takes element-wise maximum.
    This competitive activation helps suppress noise and learn robust features.
    
    Input: (B, 2*C, H, W) -> Output: (B, C, H, W)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Split channels in half
        out1, out2 = torch.chunk(x, 2, dim=1)
        # Element-wise maximum
        return torch.max(out1, out2)


class MaxFeatureMap1D(nn.Module):
    """
    Max-Feature-Map activation for 1D features (FC layers).
    Splits features in half and takes element-wise maximum.
    
    Input: (B, 2*C) -> Output: (B, C)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Split features in half
        out1, out2 = torch.chunk(x, 2, dim=1)
        # Element-wise maximum
        return torch.max(out1, out2)


class LCNNBlock(nn.Module):
    """
    LCNN convolutional block with MFM activation.
    Conv2D -> MFM -> BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Output 2*out_channels so MFM produces out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding, bias=False)
        self.mfm = MaxFeatureMap2D()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.mfm(x)
        x = self.bn(x)
        return x


class LCNNResBlock(nn.Module):
    """
    LCNN Residual block with MFM activation.
    Adds skip connection for better gradient flow.
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size, 1, padding, bias=False)
        self.mfm1 = MaxFeatureMap2D()
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size, 1, padding, bias=False)
        self.mfm2 = MaxFeatureMap2D()
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.bn1(self.mfm1(self.conv1(x)))
        out = self.bn2(self.mfm2(self.conv2(out)))
        return out + identity


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling layer.
    Computes attention-weighted mean and standard deviation.
    """
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_dim, 1, kernel_size=1),
        )
    
    def forward(self, x, eps=1e-12):
        """
        Args:
            x: (B, C, T) feature sequence
        Returns:
            (B, C*2) concatenated mean and std
        """
        # Compute attention weights
        w = self.attention(x)  # (B, 1, T)
        w = F.softmax(w, dim=2)
        
        # Weighted mean
        mu = (x * w).sum(dim=2)  # (B, C)
        
        # Weighted standard deviation
        var = ((x ** 2) * w).sum(dim=2) - mu ** 2
        sigma = torch.sqrt(torch.clamp(var, min=eps))  # (B, C)
        
        return torch.cat([mu, sigma], dim=1)  # (B, C*2)


class LCNNBackbone(nn.Module):
    """
    LCNN backbone for feature extraction.
    
    Architecture:
    - Initial conv block
    - 4 stages with increasing channels
    - Each stage has conv blocks + residual blocks + pooling
    """
    def __init__(self, channels=[32, 48, 64, 32], use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Initial convolution
        self.conv0 = LCNNBlock(1, channels[0], kernel_size=5, stride=1, padding=2)
        self.pool0 = nn.MaxPool2d(2, 2)
        
        # Stage 1
        self.conv1 = LCNNBlock(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.res1 = LCNNResBlock(channels[1]) if use_residual else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Stage 2
        self.conv2 = LCNNBlock(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        self.res2 = LCNNResBlock(channels[2]) if use_residual else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3
        self.conv3 = LCNNBlock(channels[2], channels[3], kernel_size=3, stride=1, padding=1)
        self.res3 = LCNNResBlock(channels[3]) if use_residual else nn.Identity()
        
        # Stage 4 - 1x1 convolution for channel reduction
        self.conv4 = LCNNBlock(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
        
        self.out_channels = channels[3]
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input spectrogram
        Returns:
            (B, C, H', W') feature maps
        """
        # Initial stage
        x = self.pool0(self.conv0(x))
        
        # Stage 1
        x = self.pool1(self.res1(self.conv1(x)))
        
        # Stage 2  
        x = self.pool2(self.res2(self.conv2(x)))
        
        # Stage 3
        x = self.res3(self.conv3(x))
        
        # Stage 4
        x = self.conv4(x)
        
        return x


class Model(nn.Module):
    """
    LCNN model for mel spectrogram input (--feature 1).
    
    This model uses:
    - LCNN backbone with MFM activations
    - Attentive statistics pooling for temporal aggregation
    - Classifier head with dropout regularization
    
    Input: (B, n_mels, frames) or (B, 1, n_mels, frames)
    Output: embeddings (B, emb_dim), logits (B, 2)
    """
    def __init__(self, d_args):
        super().__init__()
        
        # Model configuration
        channels = d_args.get("channels", [32, 48, 64, 32])
        dropout = d_args.get("dropout", 0.3)
        use_residual = d_args.get("use_residual", True)
        att_bottleneck = d_args.get("att_bottleneck", 64)
        emb_dim = d_args.get("emb_dim", 128)
        
        # Backbone
        self.backbone = LCNNBackbone(channels=channels, use_residual=use_residual)
        
        # Attentive pooling (operates on time axis)
        self.pool = AttentiveStatisticsPooling(self.backbone.out_channels, att_bottleneck)
        
        # Embedding dimension after pooling: out_channels * 2 (mean + std)
        pool_out_dim = self.backbone.out_channels * 2
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(pool_out_dim, emb_dim * 2),  # *2 for MFM
            MaxFeatureMap1D(),
            nn.BatchNorm1d(emb_dim),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 256),
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
        
        for m in self.embedding.modules():
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
            embeddings: (B, emb_dim) feature embeddings
            output: (B, 2) logits for classification
        """
        # Ensure 4D input: (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, n_mels, frames) -> (B, 1, n_mels, frames)
        
        # Extract features with backbone
        features = self.backbone(x)  # (B, C, H', W')
        
        # Collapse frequency axis, keep time axis for pooling
        # Mean over frequency (height) dimension
        features = features.mean(dim=2)  # (B, C, T)
        
        # Attentive statistics pooling
        pooled = self.pool(features)  # (B, C*2)
        
        # Project to embedding space
        embeddings = self.embedding(pooled)  # (B, emb_dim)
        
        # Classification
        output = self.classifier(embeddings)  # (B, 2)
        
        return embeddings, output


class ModelLarge(nn.Module):
    """
    LCNN-Large model with deeper architecture.
    Suitable for larger datasets and higher capacity requirements.
    
    Input: (B, n_mels, frames) or (B, 1, n_mels, frames)
    Output: embeddings (B, emb_dim), logits (B, 2)
    """
    def __init__(self, d_args):
        super().__init__()
        
        # Model configuration - larger channels
        channels = d_args.get("channels", [64, 96, 128, 64])
        dropout = d_args.get("dropout", 0.4)
        use_residual = d_args.get("use_residual", True)
        att_bottleneck = d_args.get("att_bottleneck", 128)
        emb_dim = d_args.get("emb_dim", 256)
        
        # Backbone
        self.backbone = LCNNBackbone(channels=channels, use_residual=use_residual)
        
        # Attentive pooling
        self.pool = AttentiveStatisticsPooling(self.backbone.out_channels, att_bottleneck)
        
        # Embedding dimension after pooling
        pool_out_dim = self.backbone.out_channels * 2
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(pool_out_dim, emb_dim * 2),
            MaxFeatureMap1D(),
            nn.BatchNorm1d(emb_dim),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 512),
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
        
        for m in self.embedding.modules():
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
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = self.backbone(x)
        features = features.mean(dim=2)
        pooled = self.pool(features)
        embeddings = self.embedding(pooled)
        output = self.classifier(embeddings)
        
        return embeddings, output
