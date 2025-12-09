"""
SimpleCNN model for ASVspoof detection
A lightweight CNN-based architecture for spoofing detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        
        self.d_args = d_args
        nb_samp = d_args.get("nb_samp", 64600)
        
        # Input: (batch_size, 1, nb_samp)
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=80, 
                               stride=4, 
                               padding=38)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(4)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(4)
        
        # Third conv layer
        self.conv3 = nn.Conv1d(in_channels=64, 
                               out_channels=128, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(4)
        
        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x, Freq_aug=False):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, nb_samp)
            Freq_aug: frequency augmentation flag (not used in SimpleCNN)
        Returns:
            embeddings: feature embeddings
            output: logits for classification
        """
        # Add channel dimension: (batch_size, 1, nb_samp)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Embeddings before classification
        embeddings = x.clone()
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return embeddings, output