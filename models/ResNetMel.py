import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class BasicBlockSE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(planes, reduction)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class ResNetMelBackbone(nn.Module):
    def __init__(self, block, layers, channels):
        super().__init__()
        self.in_planes = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.out_dim = channels[3] * block.expansion
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x shape: (B, 1, n_mels, frames) or (B, n_mels, frames)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, C, H, T)


class AttentiveStatPool(nn.Module):
    """
    Attentive statistics pooling over time (used in speaker/anti-spoof models).
    Input: (B, C, T) -> Output: (B, C*2) concatenation of weighted mean and std.
    """
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_dim, 1, kernel_size=1),
        )

    def forward(self, x, eps=1e-12):
        # x: (B, C, T)
        w = self.att(x)  # (B, 1, T)
        w = F.softmax(w, dim=2)
        mu = (x * w).sum(dim=2)  # (B, C)
        sigma = torch.sqrt(torch.clamp((x ** 2 * w).sum(dim=2) - mu ** 2, min=eps))
        return torch.cat([mu, sigma], dim=1)  # (B, C*2)


class Model(nn.Module):
    """
    SE-ResNet + AttentiveStatPool for mel spectrogram input (--feature 1).
    Recommended: use n_mels=128, frames padded to fixed length by feature extractor.
    """
    def __init__(self, d_args):
        super().__init__()
        block_channels = d_args.get("block_channels", [64, 128, 256, 512])
        layers = d_args.get("layers", [3, 4, 6, 3])  # ResNet-34/50 style; default is stronger
        dropout = d_args.get("dropout", 0.3)
        bottleneck_att = d_args.get("att_bottleneck", 128)

        self.backbone = ResNetMelBackbone(BasicBlockSE, layers, block_channels)
        self.embedding_channels = self.backbone.out_dim  # C
        self.pool = AttentiveStatPool(self.embedding_channels, bottleneck_dim=bottleneck_att)

        emb_dim = self.embedding_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        self._init_weights()

    def _init_weights(self):
        # keep light initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, Freq_aug=False):
        # x: (B, n_mels, frames) or (B, 1, n_mels, frames)
        feats = self.backbone(x)  # (B, C, H, T)
        # collapse frequency axis (H) and keep temporal axis (T) for pooling
        # do mean over frequency axis
        seq = feats.mean(dim=2)  # (B, C, T)
        emb = self.pool(seq)     # (B, C*2)
        out = self.classifier(emb)
        return emb, out