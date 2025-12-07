
# ============================================================================
# FILE: models/aasist.py
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (GraphAttentionLayer, HtrgGraphAttentionLayer, 
                     GraphPool, SincConv, ResidualBlock)


class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention
    
    Reference: 
        Jung et al., "AASIST: Audio Anti-Spoofing using Integrated 
        Spectro-Temporal Graph Attention Networks", ICASSP 2022
    """
    
    def __init__(self, config):
        super().__init__()
        
        filts = config["filts"]
        gat_dims = config["gat_dims"]
        pool_ratios = config["pool_ratios"]
        temperatures = config["temperatures"]
        
        # Frontend
        self.conv_time = SincConv(out_channels=filts[0],
                                   kernel_size=config["first_conv"])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        
        # Dropout layers
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock(nb_filts=filts[1], first=True),
            ResidualBlock(nb_filts=filts[2]),
            ResidualBlock(nb_filts=filts[3]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4])
        )
        
        # Learnable parameters
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # GAT layers
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0],
                                               temperature=temperatures[1])
        
        # Heterogeneous GAT layers
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        
        # Pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        # Output layer
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, freq_aug=False):
        # Frontend processing
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        # Encoder
        e = self.encoder(x)
        
        # Spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        
        # Temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        
        # First inference path
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        
        # Second inference path
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        
        # Apply dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        
        # Aggregate paths
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        
        # Final aggregation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, 
                                master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return last_hidden, output
