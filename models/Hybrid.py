"""
Hybrid Model for ASVspoof Detection
Optimized for high accuracy with efficient training
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFrontEnd(nn.Module):
    """Learnable 1D front-end (large-kernel conv approximating sinc behaviour)."""
    def __init__(self, out_channels=64, kernel_size=251):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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
    def __init__(self, in_ch, out_ch, scale=4):
        super().__init__()
        width = max(out_ch // scale, 1)
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_ch, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1, bias=False)
                                     for _ in range(scale - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        
        self.conv3 = nn.Conv1d(width * scale, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch)
            )

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
            sp = self.relu(self.bns[i - 1](self.convs[i - 1](sp)))
            outputs.append(sp)
        
        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        
        return self.relu(out + residual)


class GraphAttentionLayer(nn.Module):
    """Graph attention layer"""
    def __init__(self, in_dim, out_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(out_dim, out_dim)
        
        # Projection for residual if dimensions differ
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        
        # Project input for residual connection
        residual = self.res_proj(x)
        
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.out(out)
        
        return self.norm(residual + self.dropout(out))


class AttentiveStatPool1d(nn.Module):
    def __init__(self, in_dim, bottleneck=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    def forward(self, x, eps=1e-12):
        # x: (B, C, T)
        w = self.att(x)  # (B, 1, T)
        w = F.softmax(w, dim=2)
        mu = (x * w).sum(dim=2)
        sigma = torch.sqrt(torch.clamp((x ** 2 * w).sum(dim=2) - mu ** 2, min=eps))
        return torch.cat([mu, sigma], dim=1)


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
        
        frontend_ch = d_args.get("frontend_ch", 64)
        blocks_channels = d_args.get("blocks", [64, 128, 256])
        gat_dims = d_args.get("gat_dims", [128, 64])

        self.fe = ConvFrontEnd(out_channels=frontend_ch, kernel_size=d_args.get("fe_kernel", 251))

        enc = []
        prev = frontend_ch
        for ch in blocks_channels:
            enc.append(Res2NetBlock(prev, ch))
            enc.append(nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
            prev = ch
        self.encoder = nn.Sequential(*enc)

        self.node_proj = nn.Linear(blocks_channels[-1], gat_dims[0])
        self.gat1 = GraphAttentionLayer(gat_dims[0], gat_dims[0], num_heads=2)
        self.gat2 = GraphAttentionLayer(gat_dims[0], gat_dims[1], num_heads=2)

        self.pool = AttentiveStatPool1d(gat_dims[1], bottleneck=d_args.get("att_bottleneck", 128))
        self.classifier = nn.Sequential(
            nn.Linear(gat_dims[1] * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, Freq_aug=False):
        # expects mel input shape (B, n_mels, frames) -> collapse freq to 1 and treat as 1D temporal input
        if x.dim() == 3:
            x = x.mean(dim=1, keepdim=True)  # (B,1,T)
        z = self.fe(x)               # (B, C, T)
        z = self.encoder(z)         # (B, C2, T2)
        z_t = z.transpose(1, 2)     # (B, N, C)
        nodes = self.node_proj(z_t) # (B, N, G)
        nodes = self.gat1(nodes)
        nodes = self.gat2(nodes)
        feat = nodes.transpose(1, 2)  # (B, G2, N)
        emb = self.pool(feat)         # (B, G2*2)
        out = self.classifier(emb)
        return emb, out