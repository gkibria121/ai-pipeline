"""
Transformer-based model for ASVspoof detection
Uses self-attention mechanisms to detect audio spoofing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AudioPreprocessor(nn.Module):
    """Preprocess raw audio into embeddings"""
    def __init__(self, d_model=256):
        super().__init__()
        # Conv layers to extract features from raw audio
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch_size, nb_samp)
        x = x.unsqueeze(1)  # (batch_size, 1, nb_samp)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, 
                                                     dropout=dropout, 
                                                     batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention block
        attn_out, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feedforward block
        ff_out = self.linear2(self.dropout2(F.relu(self.linear1(x))))
        x = x + self.dropout3(ff_out)
        x = self.norm2(x)
        return x


class Model(nn.Module):
    """Transformer-based ASVspoof detection model"""
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        
        # Model parameters
        d_model = d_args.get("d_model", 256)
        nhead = d_args.get("nhead", 8)
        num_layers = d_args.get("num_layers", 4)
        dim_feedforward = d_args.get("dim_feedforward", 1024)
        dropout = d_args.get("dropout", 0.1)
        
        # Audio preprocessing
        self.preprocessor = AudioPreprocessor(d_model=d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, nhead, dim_feedforward, dropout) 
              for _ in range(num_layers)]
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(d_model, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x, Freq_aug=False):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, nb_samp)
            Freq_aug: frequency augmentation flag
        Returns:
            embeddings: feature embeddings
            output: logits for classification
        """
        # Preprocess audio
        x = self.preprocessor(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer blocks
        x = self.transformer_blocks(x)
        
        # Global average pooling
        embeddings = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.dropout(embeddings)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        
        return embeddings, output