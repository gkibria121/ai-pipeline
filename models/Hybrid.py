"""
AASIST Hybrid Model for ASVspoof Detection
Combines Graph Attention Networks with Res2Net and Squeeze-Excitation modules
Better performance on ASVspoof2019 and ASVspoof2021
Hybrid architecture: SincConv + Res2Net + Multi-head GAT + SE blocks
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
        # x: (bs, channels, height, width)
        b, c, h, w = x.size()
        se = x.mean((2, 3), keepdim=True)  # Global average pooling
        se = se.view(b, c)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        se = se.view(b, c, 1, 1)
        return x * se


class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels, scale=4, stride=1):
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width * scale)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                     padding=1, dilation=1)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale - 1)])
        
        self.conv3 = nn.Conv2d(width * scale, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Split into scale groups
        xs = torch.chunk(out, self.scale, dim=1)
        ys = []
        for i, xi in enumerate(xs):
            if i == 0:
                ys.append(xi)
            else:
                xi = self.convs[i - 1](xi + xs[i - 1])
                xi = self.relu(self.bns[i - 1](xi))
                ys.append(xi)
        
        out = torch.cat(ys, dim=1)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        return out + identity


class MultiHeadGraphAttentionLayer(nn.Module):
    """Multi-head Graph Attention Layer with Layer Normalization"""
    def __init__(self, in_dim, out_dim, num_heads=4, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        # Multi-head attention
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.output = nn.Linear(out_dim, out_dim)
        
        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        """
        x: (bs, num_nodes, in_dim)
        """
        bs, num_nodes, _ = x.size()
        
        # Multi-head self-attention
        q = self.query(x).view(bs, num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(bs, num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(bs, num_nodes, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # (bs, num_heads, num_nodes, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim * self.temp)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(bs, num_nodes, self.out_dim)
        
        # Output projection
        out = self.output(context)
        out = self.dropout(out)
        out = self.norm1(x + out)
        
        # Feed-forward network
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out


class SincConv(nn.Module):
    """Mel-scaled SincConv for better frequency selectivity"""
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        if kernel_size % 2 == 0:
            self.kernel_size += 1
        
        # Mel-spaced filter banks
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = 2595 * np.log10(1 + f / 700)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, out_channels + 1)
        filbandwidthsf = 700 * (10**(filbandwidthsmel / 2595) - 1)
        
        self.register_buffer('mel_freqs', torch.FloatTensor(filbandwidthsf))
        # Register hsupp as a buffer so it moves to the correct device
        self.register_buffer('hsupp', torch.arange(-(self.kernel_size - 1) / 2, 
                                                   (self.kernel_size - 1) / 2 + 1,
                                                   dtype=torch.float32))
        
    def forward(self, x, mask=False):
        band_pass_filter = torch.zeros(self.out_channels, self.kernel_size, 
                                       device=x.device, dtype=x.dtype)
        
        # Move mel_freqs to the same device as x
        mel_freqs = self.mel_freqs.to(x.device)
        hsupp = self.hsupp.to(x.device)
        
        for i in range(len(mel_freqs) - 1):
            fmin = mel_freqs[i]
            fmax = mel_freqs[i + 1]
            
            hHigh = (2 * fmax / self.sample_rate) * torch.sinc(
                2 * fmax * hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * torch.sinc(
                2 * fmin * hsupp / self.sample_rate)
            hideal = hHigh - hLow
            
            band_pass_filter[i, :] = torch.from_numpy(np.hamming(
                self.kernel_size)).to(x.device).to(x.dtype) * hideal
        
        if mask:
            A = np.random.randint(0, 20)
            A0 = random.randint(0, band_pass_filter.shape[0] - max(1, A))
            band_pass_filter[A0:A0 + max(1, A), :] = 0
        
        filters = band_pass_filter.unsqueeze(1)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class Model(nn.Module):
    """
    AASIST Hybrid Model
    Hybrid architecture combining:
    - SincConv front-end for frequency-selective filtering
    - Res2Net encoder for multi-scale feature extraction
    - Multi-head Graph Attention for feature fusion
    - Squeeze-Excitation blocks for channel attention
    """
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        
        filts = d_args.get("filts", [128, 64, 64, 64])
        gat_dims = d_args.get("gat_dims", [64, 32])
        
        # ============================================
        # Stage 1: SincConv Front-end
        # ============================================
        self.conv_time = SincConv(out_channels=filts[0], 
                                  kernel_size=d_args.get("first_conv", 128))
        self.first_bn = nn.BatchNorm1d(filts[0])
        self.first_se = SEBlock(filts[0])
        
        # ============================================
        # Stage 2: Res2Net Encoder (Multi-scale)
        # ============================================
        self.encoder = nn.Sequential(
            Res2NetBlock(filts[0], filts[1], scale=4, stride=1),
            Res2NetBlock(filts[1], filts[1], scale=4, stride=1),
            Res2NetBlock(filts[1], filts[2], scale=4, stride=2),
            Res2NetBlock(filts[2], filts[2], scale=4, stride=1),
            Res2NetBlock(filts[2], filts[3], scale=4, stride=2),
            Res2NetBlock(filts[3], filts[3], scale=4, stride=1),
        )
        
        # ============================================
        # Stage 3: Multi-head Graph Attention
        # ============================================
        self.gat_spectral = MultiHeadGraphAttentionLayer(
            filts[3], gat_dims[0], num_heads=4)
        self.gat_temporal = MultiHeadGraphAttentionLayer(
            filts[3], gat_dims[0], num_heads=4)
        self.gat_fused = MultiHeadGraphAttentionLayer(
            gat_dims[0], gat_dims[1], num_heads=4)
        
        # ============================================
        # Stage 4: Adaptive Pooling
        # ============================================
        self.adaptive_pool_spectral = nn.AdaptiveAvgPool1d(1)
        self.adaptive_pool_temporal = nn.AdaptiveAvgPool1d(1)
        
        # ============================================
        # Stage 5: Classification Head
        # ============================================
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(gat_dims[1] * 4, 2)

    def forward(self, x, Freq_aug=False):
        """
        Forward pass
        Args:
            x: (bs, nb_samp) - input audio samples
            Freq_aug: bool - apply frequency augmentation
        Returns:
            embeddings: (bs, gat_dims[1]*4) - feature embeddings
            output: (bs, 2) - classification logits
        """
        # ============================================
        # Stage 1: SincConv Front-end
        # ============================================
        x = x.unsqueeze(1)  # (bs, 1, nb_samp)
        x = self.conv_time(x, mask=Freq_aug)  # (bs, filts[0], time)
        x = self.first_bn(x)
        x = F.relu(x)
        x = x.unsqueeze(1)  # (bs, 1, filts[0], time)
        
        # ============================================
        # Stage 2: Res2Net Encoder
        # ============================================
        x = self.encoder(x)  # (bs, filts[3], freq, time)
        
        # ============================================
        # Stage 3a: Spectral Processing
        # ============================================
        x_spectral, _ = torch.max(torch.abs(x), dim=3)  # (bs, filts[3], freq)
        x_spectral = x_spectral.transpose(1, 2)  # (bs, freq, filts[3])
        gat_spectral = self.gat_spectral(x_spectral)  # (bs, freq, gat_dims[0])
        pool_spectral = self.adaptive_pool_spectral(
            gat_spectral.transpose(1, 2))  # (bs, gat_dims[0], 1)
        pool_spectral = pool_spectral.squeeze(-1)
        
        # ============================================
        # Stage 3b: Temporal Processing
        # ============================================
        x_temporal, _ = torch.max(torch.abs(x), dim=2)  # (bs, filts[3], time)
        x_temporal = x_temporal.transpose(1, 2)  # (bs, time, filts[3])
        gat_temporal = self.gat_temporal(x_temporal)  # (bs, time, gat_dims[0])
        pool_temporal = self.adaptive_pool_temporal(
            gat_temporal.transpose(1, 2))  # (bs, gat_dims[0], 1)
        pool_temporal = pool_temporal.squeeze(-1)
        
        # ============================================
        # Stage 3c: Fused Representation
        # ============================================
        x_fused = torch.cat([gat_spectral, gat_temporal], 
                           dim=1)  # (bs, freq+time, gat_dims[0])
        gat_fused = self.gat_fused(x_fused)  # (bs, freq+time, gat_dims[1])
        pool_fused = gat_fused.mean(dim=1)  # (bs, gat_dims[1])
        
        # ============================================
        # Stage 4: Feature Aggregation
        # ============================================
        embeddings = torch.cat([pool_spectral, pool_temporal, 
                               pool_fused, pool_fused], dim=1)
        embeddings = self.dropout(embeddings)
        
        # ============================================
        # Stage 5: Classification
        # ============================================
        output = self.output_layer(embeddings)
        
        return embeddings, output

