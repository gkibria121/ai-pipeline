import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000):
        super().__init__()
        kernel_size += 1 - kernel_size % 2
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size

        low_hz = 30
        high_hz = sample_rate / 2 - (sample_rate / 2 - low_hz) / (out_channels + 1)

        mel_low = 2595 * math.log10(1 + low_hz / 700)
        mel_high = 2595 * math.log10(1 + high_hz / 700)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        self.low_hz = nn.Parameter(hz_points[:-1].view(-1, 1))
        self.band_hz = nn.Parameter(torch.diff(hz_points).view(-1, 1))

        n = torch.arange(kernel_size).float()
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))
        self.register_buffer("window", window)
        self.register_buffer("n_", (n - (kernel_size - 1) / 2) / sample_rate)

    def forward(self, x):
        low = torch.abs(self.low_hz) + 1.0
        high = torch.clamp(low + torch.abs(self.band_hz), max=self.sample_rate / 2)

        f_low = low / self.sample_rate
        f_high = high / self.sample_rate

        band_pass = 2 * f_high * torch.sinc(2 * f_high * self.sample_rate * self.n_) - \
                    2 * f_low * torch.sinc(2 * f_low * self.sample_rate * self.n_)
        band_pass = band_pass * self.window
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)

        return F.conv1d(x, band_pass.unsqueeze(1), padding=self.kernel_size // 2)


class Res2NetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=4):
        super().__init__()
        width = out_ch // scale
        self.scale = scale
        self.width = width

        self.conv1 = nn.Conv1d(in_ch, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
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
            sp = spx[i] if i == 1 else sp + spx[i]
            sp = self.relu(self.bns[i - 1](self.convs[i - 1](sp)))
            outputs.append(sp)

        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.relu(out + residual)
        return out


class RawNet3Backbone(nn.Module):
    def __init__(self, nb_samp=64600, channels=[64, 128, 256, 512]):
        super().__init__()
        self.nb_samp = nb_samp
        self.sinc = SincConv(out_channels=channels[0], kernel_size=251)

        blocks = []
        in_ch = channels[0]
        for ch in channels:
            blocks.append(
                nn.Sequential(
                    Res2NetBlock(in_ch, ch),
                    nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
                )
            )
            in_ch = ch
        self.encoder = nn.Sequential(*blocks)

        self.attention = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1] // 4, 1),
            nn.Tanh(),
            nn.Conv1d(channels[-1] // 4, channels[-1], 1),
            nn.Softmax(dim=-1)
        )

        self.fc = nn.Linear(channels[-1], channels[-1])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.sinc(x)
        x = torch.abs(x)
        x = self.encoder(x)

        attn = self.attention(x)
        pooled = (x * attn).sum(dim=-1)
        emb = self.fc(pooled)
        emb = F.dropout(emb, p=0.2, training=self.training)
        return emb


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        nb_samp = d_args.get("nb_samp", 64600)
        channels = d_args.get("channels", [64, 128, 256, 512])
        self.backbone = RawNet3Backbone(nb_samp, channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x, Freq_aug=False):
        emb = self.backbone(x)
        out = self.classifier(emb)
        return emb, out