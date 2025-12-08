"""
AASIST Hybrid Model for ASVspoof Detection
Simplified architecture that properly handles tensor dimensions
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SEBlock(nn.Module):
    """Squeeze-Excitation Block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        se = x.mean(dim=-1)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se.unsqueeze(-1)


class ConvBlock(nn.Module):
    """Basic Conv Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return x


class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        return out


class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention"""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.fc_out = nn.Linear(out_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """x: (bs, seq_len, in_dim)"""
        bs, seq_len, _ = x.size()
        
        q = self.query(x).view(bs, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(bs, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(bs, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(bs, seq_len, self.out_dim)
        
        out = self.fc_out(context)
        out = self.dropout(out)
        out = self.norm(x + out)
        
        return out


class SincConv(nn.Module):
    """SincConv with Mel-scale"""
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate
        
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = 2595 * np.log10(1 + f / 700)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, out_channels + 1)
        filbandwidthsf = 700 * (10**(filbandwidthsmel / 2595) - 1)
        
        self.register_buffer('mel_freqs', torch.FloatTensor(filbandwidthsf))
        self.register_buffer('hsupp', torch.arange(-(self.kernel_size - 1) / 2, 
                                                   (self.kernel_size - 1) / 2 + 1,
                                                   dtype=torch.float32))

    def forward(self, x, mask=False):
        band_pass_filter = torch.zeros(self.out_channels, self.kernel_size, 
                                       device=x.device, dtype=x.dtype)
        
        mel_freqs = self.mel_freqs.to(x.device).to(x.dtype)
        hsupp = self.hsupp.to(x.device).to(x.dtype)
        
        for i in range(len(mel_freqs) - 1):
            fmin = mel_freqs[i]
            fmax = mel_freqs[i + 1]
            
            hHigh = (2 * fmax / self.sample_rate) * torch.sinc(
                2 * fmax * hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * torch.sinc(
                2 * fmin * hsupp / self.sample_rate)
            hideal = hHigh - hLow
            
            band_pass_filter[i, :] = torch.from_numpy(
                np.hamming(self.kernel_size)).to(x.device).to(x.dtype) * hideal
        
        if mask:
            A = np.random.randint(0, 20)
            A0 = random.randint(0, max(1, band_pass_filter.shape[0] - A))
            band_pass_filter[A0:A0 + A, :] = 0
        
        filters = band_pass_filter.unsqueeze(1)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class Model(nn.Module):
    """
    Simplified AASIST Hybrid Model
    Architecture: SincConv → Residual Blocks → GAT → Classification
    """
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        
        filts = d_args.get("filts", [128, 64, 64, 64])
        gat_dims = d_args.get("gat_dims", [64, 32])
        
        # ============================================
        # Stage 1: SincConv Front-end
        # ============================================
        self.sinc = SincConv(out_channels=filts[0], 
                            kernel_size=d_args.get("first_conv", 128))
        self.bn_sinc = nn.BatchNorm1d(filts[0])
        
        # ============================================
        # Stage 2: Residual Encoder
        # ============================================
        self.res_blocks = nn.Sequential(
            ResBlock(filts[0], filts[1], stride=1),
            ResBlock(filts[1], filts[1], stride=1),
            ResBlock(filts[1], filts[2], stride=2),
            ResBlock(filts[2], filts[2], stride=1),
            ResBlock(filts[2], filts[3], stride=2),
            ResBlock(filts[3], filts[3], stride=1),
        )
        
        # ============================================
        # Stage 3: Global and Local pooling
        # ============================================
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(8)
        
        # ============================================
        # Stage 4: Graph Attention
        # ============================================
        local_dim = filts[3] * 8
        self.gat = MultiHeadGraphAttention(filts[3], gat_dims[0], num_heads=4)
        
        # ============================================
        # Stage 5: Classification Head
        # ============================================
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(gat_dims[0] + filts[3], 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, Freq_aug=False):
        """
        x: (bs, nb_samp)
        """
        # ============================================
        # Stage 1: SincConv
        # ============================================
        x = x.unsqueeze(1)  # (bs, 1, nb_samp)
        x = self.sinc(x, mask=Freq_aug)  # (bs, filts[0], time)
        x = self.bn_sinc(x)
        x = F.relu(x)
        
        # ============================================
        # Stage 2: Residual Encoder
        # ============================================
        x = self.res_blocks(x)  # (bs, filts[3], time)
        
        # ============================================
        # Stage 3: Pooling
        # ============================================
        # Global pooling
        global_feat = self.global_pool(x).squeeze(-1)  # (bs, filts[3])
        
        # Local pooling for GAT
        local_feat = self.local_pool(x)  # (bs, filts[3], 8)
        local_feat = local_feat.transpose(1, 2)  # (bs, 8, filts[3])
        
        # ============================================
        # Stage 4: Graph Attention
        # ============================================
        gat_feat = self.gat(local_feat)  # (bs, 8, gat_dims[0])
        gat_feat = gat_feat.mean(dim=1)  # (bs, gat_dims[0])
        
        # ============================================
        # Stage 5: Feature Fusion & Classification
        # ============================================
        embeddings = torch.cat([global_feat, gat_feat], dim=1)  # (bs, filts[3] + gat_dims[0])
        embeddings = self.dropout(embeddings)
        
        x = F.relu(self.fc1(embeddings))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return embeddings, output

