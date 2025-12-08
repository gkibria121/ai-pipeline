"""
Hybrid Model for ASVspoof Detection
Optimized for high accuracy with efficient training
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
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
            band_pass = band_pass.clone()
            band_pass[mask_indices, :] = 0
        
        return F.conv1d(x, band_pass.unsqueeze(1), padding=self.kernel_size // 2)


class SEBlock(nn.Module):
    """Squeeze-Excitation attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.fc(x).unsqueeze(-1)
        return x * weight


class Res2NetBlock(nn.Module):
    """Multi-scale residual block"""
    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()
        width = max(out_channels // scale, 1)
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(scale - 1):
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
        
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        spx = torch.split(out, self.width, dim=1)
        outputs = [spx[0]]
        
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i-1](self.convs[i-1](sp)))
            outputs.append(sp)
        
        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        return self.relu(out + residual)


class GraphAttentionLayer(nn.Module):
    """Graph attention layer"""
    def __init__(self, in_dim, out_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # Projection for residual if dimensions differ
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        
        # Project input for residual connection
        residual = self.res_proj(x)
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)
        
        return self.norm(residual + self.dropout(out))


class CrossAttention(nn.Module):
    """Cross-attention between spectral and temporal features"""
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # Cross attention
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm(query + attn_out)
        # FFN
        query = self.norm2(query + self.ffn(query))
        return query


class MaxGraphPool(nn.Module):
    """Max-graph pooling"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, N, dim)
        gate = torch.sigmoid(self.proj(x))  # (B, N, out_dim)
        x_expand = x.unsqueeze(-1)  # (B, N, dim, 1)
        gate_expand = gate.unsqueeze(-2)  # (B, N, 1, out_dim)
        gated = x_expand * gate_expand  # (B, N, dim, out_dim)
        pooled = gated.max(dim=1)[0]  # (B, dim, out_dim)
        return pooled.mean(dim=1)  # (B, out_dim)


class Model(nn.Module):
    """
    Hybrid Model for ASVspoof Detection
    
    Architecture:
    1. SincConv frontend with frequency masking
    2. Dual-path encoder (Spectral + Temporal)
    3. Graph attention with cross-attention
    4. Max-graph pooling
    5. Classification head
    """
    def __init__(self, d_args):
        super().__init__()
        
        filts = d_args.get("filts", [70, 64, 64, 64])
        gat_dims = d_args.get("gat_dims", [64, 32])
        first_conv = d_args.get("first_conv", 129)
        
        # ============================================
        # Stage 1: SincConv Frontend
        # ============================================
        self.sinc = SincConv(out_channels=filts[0], kernel_size=first_conv)
        self.bn0 = nn.BatchNorm1d(filts[0])
        self.selu = nn.SELU(inplace=True)
        
        # ============================================
        # Stage 2: Spectral Encoder (Res2Net)
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
            nn.Conv1d(filts[0], filts[1], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(filts[1]),
            nn.SELU(inplace=True),
            nn.Conv1d(filts[1], filts[2], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(filts[2]),
            nn.SELU(inplace=True),
            nn.Conv1d(filts[2], filts[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(filts[3]),
            nn.SELU(inplace=True),
        )
        
        # ============================================
        # Stage 4: Graph Attention
        # ============================================
        self.pool_size = 16
        self.spec_pool = nn.AdaptiveAvgPool1d(self.pool_size)
        self.temp_pool = nn.AdaptiveAvgPool1d(self.pool_size)
        
        # GAT layers with proper dimension handling
        self.gat1 = GraphAttentionLayer(filts[3], gat_dims[0], num_heads=2)
        self.cross_attn = CrossAttention(gat_dims[0], num_heads=2)
        self.gat2 = GraphAttentionLayer(gat_dims[0], gat_dims[1], num_heads=2)
        
        # Projection for temporal features to match GAT output
        self.temp_proj = nn.Linear(filts[3], gat_dims[0])
        
        # ============================================
        # Stage 5: Pooling
        # ============================================
        self.max_graph_pool = MaxGraphPool(gat_dims[1], gat_dims[1])
        
        # ============================================
        # Stage 6: Classifier
        # ============================================
        classifier_in = gat_dims[1] + filts[3] * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.BatchNorm1d(128),
            nn.SELU(inplace=True),
            nn.Dropout(0.5),
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
        # ============================================
        # Stage 1: SincConv Frontend
        # ============================================
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc(x, mask=Freq_aug)  # (B, filts[0], T)
        x = self.selu(self.bn0(x))
        
        # ============================================
        # Stage 2 & 3: Dual-path Encoding
        # ============================================
        spec_feat = self.spec_encoder(x)  # (B, filts[3], T1)
        temp_feat = self.temp_encoder(x)  # (B, filts[3], T2)
        
        # ============================================
        # Stage 4: Graph Attention
        # ============================================
        spec_seq = self.spec_pool(spec_feat).transpose(1, 2)  # (B, 16, filts[3])
        temp_seq = self.temp_pool(temp_feat).transpose(1, 2)  # (B, 16, filts[3])
        
        # Self-attention on spectral
        spec_seq = self.gat1(spec_seq)  # (B, 16, gat_dims[0])
        
        # Project temporal for cross-attention
        temp_proj = self.temp_proj(temp_seq)  # (B, 16, gat_dims[0])
        
        # Cross-attention
        spec_seq = self.cross_attn(spec_seq, temp_proj)  # (B, 16, gat_dims[0])
        
        # Final GAT
        spec_seq = self.gat2(spec_seq)  # (B, 16, gat_dims[1])
        
        # ============================================
        # Stage 5: Pooling & Classification
        # ============================================
        graph_out = self.max_graph_pool(spec_seq)  # (B, gat_dims[1])
        global_avg = spec_feat.mean(dim=-1)  # (B, filts[3])
        global_max = spec_feat.max(dim=-1)[0]  # (B, filts[3])
        
        embeddings = torch.cat([graph_out, global_avg, global_max], dim=1)
        output = self.classifier(embeddings)
        
        return embeddings, output