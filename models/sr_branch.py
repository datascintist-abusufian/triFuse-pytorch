import torch
import torch.nn as nn
import torch.nn.functional as F


class DilationBlock(nn.Module):
    """Residual block with dilated convolution."""
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SRBranch(nn.Module):
    """
    Structural Recovery Expert.
    Aggregates all encoder levels and uses dilated convolutions to restore weak responses.
    """
    
    def __init__(self, skip_channels=[64, 128, 256, 512, 1024], 
                 projected_width=256, num_blocks=4, dilation_schedule=[1, 2, 4, 8]):
        super().__init__()
        
        self.projected_width = projected_width
        
        # Project all encoder features to unified embedding space
        self.projections = nn.ModuleList()
        for ch in skip_channels:
            self.projections.append(
                nn.Conv2d(ch, projected_width, 1)
            )
        
        # SR encoder with dilated convolutions
        self.sr_blocks = nn.ModuleList()
        in_ch = projected_width
        
        for i in range(num_blocks):
            dilation = dilation_schedule[i % len(dilation_schedule)]
            block = DilationBlock(in_ch, projected_width, dilation)
            self.sr_blocks.append(block)
            in_ch = projected_width
        
        # SR decoder (simpler than CNN expert)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(projected_width, projected_width // 2, 3, 1, 1),
            nn.BatchNorm2d(projected_width // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(projected_width // 2, 3, 1)  # 3 classes
        )
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List [F_0, F_1, F_2, F_3, F_4]
        Returns:
            Logits for 3 classes (before HMNA refinement)
        """
        # Project and align all levels
        projected = []
        target_size = encoder_features[-1].shape[2:]  # Smallest feature map
        
        for i, feat in enumerate(encoder_features):
            p = self.projections[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        
        # Aggregate by concatenation
        aggregated = torch.cat(projected, dim=1)  # (B, projected_width * 5, H, W)
        aggregated = nn.Conv2d(len(encoder_features) * self.projected_width, 
                               self.projected_width, 1)(aggregated)
        
        # Apply SR blocks with dilation
        x = aggregated
        for block in self.sr_blocks:
            x = block(x)
        
        # Decode
        logits = self.decoder(x)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return logits
