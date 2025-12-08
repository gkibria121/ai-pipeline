"""
HybridV3: Improved model addressing overfitting and attack-type weakness
Key improvements:
1. Stronger regularization
2. Multi-scale temporal modeling
3. Attack-invariant feature learning
4. Proper heterogeneous attention
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SincConvFast(nn.Module):
    """Learnable SincConv filters"""
    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate
        
        low_hz = 30
        high_hz = sample_rate / 2 - (sample_rate / 2 - low_hz) / (out_channels + 1)
        
        mel_low = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))
        
        n = torch.linspace(0, self.kernel_size - 1, self.kernel_size)
        self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * np.pi * n / self.kernel_size))
        
        n_half = (self.kernel_size - 1) / 2
        self.register_buffer('n_', (torch.arange(self.kernel_size).float() - n_half) / self.sample_rate)

    def forward(self, x, mask=False):
        low = torch.abs(self.low_hz_) + 1.0
        high = torch.clamp(low + torch.abs(self.band_hz_), min=1.0, max=self.sample_rate / 2)
        
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        
        band_pass = 2 * f_high * torch.sinc(2 * f_high * self.sample_rate * self.n_) - \
                    2 * f_low * torch.sinc(2 * f_low * self.sample_rate * self.n_)
        
        band_pass = band_pass * self.window
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)
        
        if mask and self.training:
            num_mask = random.randint(0, 10)
            mask_indices = random.sample(range(self.out_channels), min(num_mask, self.out_channels))
            band_pass[mask_indices, :] = 0
        
        return F.conv1d(x, band_pass.unsqueeze(1), padding=self.kernel_size // 2)


class Res2NetBlock(nn.Module):
    """Multi-scale residual block (from AASIST)"""
    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()
        width = out_channels // scale
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(width, width, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm1d(width))
        
        self.conv3 = nn.Conv1d(width * scale, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        # SE attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        spx = torch.split(out, self.width, dim=1)
        outputs = []
        sp = spx[0]
        outputs.append(sp)
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i-1](self.convs[i-1](sp)))
            outputs.append(sp)
        
        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        
        # SE attention
        se_weight = self.se(out).unsqueeze(-1)
        out = out * se_weight
        
        out = out + residual
        return self.relu(out)


class HeterogeneousGraphAttention(nn.Module):
    """
    Heterogeneous Graph Attention for spectral and temporal features
    Similar to AASIST's approach
    """
    def __init__(self, in_dim, out_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Spectral attention
        self.spec_q = nn.Linear(in_dim, out_dim)
        self.spec_k = nn.Linear(in_dim, out_dim)
        self.spec_v = nn.Linear(in_dim, out_dim)
        
        # Temporal attention
        self.temp_q = nn.Linear(in_dim, out_dim)
        self.temp_k = nn.Linear(in_dim, out_dim)
        self.temp_v = nn.Linear(in_dim, out_dim)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=0.1, batch_first=True)
        
        self.out_proj = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def _attention(self, q, k, v):
        B, N, _ = q.shape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return out

    def forward(self, spec_feat, temp_feat):
        """
        spec_feat: (B, N_spec, dim) - spectral features
        temp_feat: (B, N_temp, dim) - temporal features
        """
        # Self attention on spectral
        spec_q = self.spec_q(spec_feat)
        spec_k = self.spec_k(spec_feat)
        spec_v = self.spec_v(spec_feat)
        spec_out = self._attention(spec_q, spec_k, spec_v)
        
        # Self attention on temporal
        temp_q = self.temp_q(temp_feat)
        temp_k = self.temp_k(temp_feat)
        temp_v = self.temp_v(temp_feat)
        temp_out = self._attention(temp_q, temp_k, temp_v)
        
        # Cross attention (spectral attends to temporal)
        cross_out, _ = self.cross_attn(spec_out, temp_out, temp_out)
        
        # Combine
        combined = torch.cat([spec_out, cross_out], dim=-1)
        out = self.out_proj(combined)
        out = self.norm(spec_feat + self.dropout(out))
        
        return out


class MaxGraphPool(nn.Module):
    """Max-graph pooling (from AASIST)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, N, dim)
        gate = self.sigmoid(self.proj(x))  # (B, N, out_dim)
        x_gated = x.unsqueeze(-1) * gate.unsqueeze(-2)  # (B, N, dim, out_dim)
        out = x_gated.max(dim=1)[0]  # (B, dim, out_dim)
        return out.mean(dim=1)  # (B, out_dim)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) for regularization"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Model(nn.Module):
    """
    HybridV3: High-performance model targeting 98% accuracy
    
    Key features:
    1. SincConv with frequency masking
    2. Res2Net multi-scale encoder
    3. Heterogeneous graph attention (spectral + temporal)
    4. Max-graph pooling
    5. Strong regularization (dropout, drop path, weight decay)
    """
    def __init__(self, d_args):
        super().__init__()
        
        filts = d_args.get("filts", [70, 64, 64, 64])
        gat_dims = d_args.get("gat_dims", [64, 32])
        first_conv = d_args.get("first_conv", 129)
        
        # ============================================
        # Stage 1: SincConv Frontend
        # ============================================
        self.sinc = SincConvFast(out_channels=filts[0], kernel_size=first_conv)
        self.bn_sinc = nn.BatchNorm1d(filts[0])
        self.selu = nn.SELU(inplace=True)
        
        # ============================================
        # Stage 2: Res2Net Encoder (Spectral Path)
        # ============================================
        self.spec_encoder = nn.Sequential(
            Res2NetBlock(filts[0], filts[1]),
            nn.MaxPool1d(3),
            Res2NetBlock(filts[1], filts[2]),
            nn.MaxPool1d(3),
            Res2NetBlock(filts[2], filts[3]),
            nn.MaxPool1d(3),
        )
        
        # ============================================
        # Stage 3: Temporal Encoder
        # ============================================
        self.temp_encoder = nn.Sequential(
            nn.Conv1d(filts[0], filts[1], 7, stride=2, padding=3),
            nn.BatchNorm1d(filts[1]),
            nn.SELU(inplace=True),
            nn.Conv1d(filts[1], filts[2], 5, stride=2, padding=2),
            nn.BatchNorm1d(filts[2]),
            nn.SELU(inplace=True),
            nn.Conv1d(filts[2], filts[3], 3, stride=2, padding=1),
            nn.BatchNorm1d(filts[3]),
            nn.SELU(inplace=True),
        )
        
        # ============================================
        # Stage 4: Heterogeneous Graph Attention
        # ============================================
        self.spec_pool = nn.AdaptiveAvgPool1d(16)
        self.temp_pool = nn.AdaptiveAvgPool1d(16)
        
        self.hgat1 = HeterogeneousGraphAttention(filts[3], gat_dims[0], num_heads=2)
        self.hgat2 = HeterogeneousGraphAttention(gat_dims[0], gat_dims[1], num_heads=2)
        
        self.drop_path = DropPath(0.1)
        
        # ============================================
        # Stage 5: Graph Pooling
        # ============================================
        self.max_pool_spec = MaxGraphPool(gat_dims[1], gat_dims[1])
        self.max_pool_temp = MaxGraphPool(filts[3], gat_dims[1])
        
        # Global stats
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # ============================================
        # Stage 6: Classification Head
        # ============================================
        # Features: max_pool_spec + max_pool_temp + global_avg + global_max
        classifier_in = gat_dims[1] * 2 + filts[3] * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.BatchNorm1d(128),
            nn.SELU(inplace=True),
            nn.Dropout(0.5),  # Strong dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, Freq_aug=False):
        """
        Args:
            x: (B, T) raw audio
            Freq_aug: frequency masking augmentation
        Returns:
            embeddings, output
        """
        # ============================================
        # Stage 1: SincConv
        # ============================================
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc(x, mask=Freq_aug)  # (B, filts[0], T)
        x = self.selu(self.bn_sinc(x))
        
        # ============================================
        # Stage 2 & 3: Parallel Encoders
        # ============================================
        spec_feat = self.spec_encoder(x)  # (B, filts[3], T1)
        temp_feat = self.temp_encoder(x)  # (B, filts[3], T2)
        
        # ============================================
        # Stage 4: Heterogeneous Graph Attention
        # ============================================
        spec_feat_pooled = self.spec_pool(spec_feat).transpose(1, 2)  # (B, 16, filts[3])
        temp_feat_pooled = self.temp_pool(temp_feat).transpose(1, 2)  # (B, 16, filts[3])
        
        # HGAT layers
        gat_out = self.hgat1(spec_feat_pooled, temp_feat_pooled)
        gat_out = self.drop_path(gat_out)
        gat_out = self.hgat2(gat_out, temp_feat_pooled)
        
        # ============================================
        # Stage 5: Pooling & Aggregation
        # ============================================
        # Max-graph pooling
        spec_graph = self.max_pool_spec(gat_out)  # (B, gat_dims[1])
        temp_graph = self.max_pool_temp(temp_feat_pooled)  # (B, gat_dims[1])
        
        # Global statistics
        global_avg = self.global_avg(spec_feat).squeeze(-1)  # (B, filts[3])
        global_max = self.global_max(spec_feat).squeeze(-1)  # (B, filts[3])
        
        # ============================================
        # Stage 6: Classification
        # ============================================
        embeddings = torch.cat([spec_graph, temp_graph, global_avg, global_max], dim=1)
        output = self.classifier(embeddings)
        
        return embeddings, output