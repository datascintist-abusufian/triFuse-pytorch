import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class SharedEncoder(nn.Module):
    """
    Shared convolutional encoder with four-stage hierarchical feature extraction.
    
    Produces features F_0 through F_4 at different resolutions.
    """
    
    def __init__(self, in_channels=1, embed_dim=64, depths=[2, 2, 3, 3], 
                 channels=[64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Initial embedding
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Four encoder stages
        self.stages = nn.ModuleList()
        in_ch = embed_dim
        
        for i, (out_ch, depth) in enumerate(zip(channels[:4], depths)):
            stage = nn.ModuleList()
            
            # First block may change resolution
            if i == 0:
                stage.append(ResidualBlock(in_ch, out_ch, stride=1))
            else:
                stage.append(ResidualBlock(in_ch, out_ch, stride=2))
            
            # Remaining blocks
            for _ in range(depth - 1):
                stage.append(ResidualBlock(out_ch, out_ch, stride=1))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # Store channel dimensions
        self.channels = [embed_dim] + channels[:4]
    
    def forward(self, x):
        """
        Returns:
            List of features [F_0, F_1, F_2, F_3, F_4]
            where F_l has shape (B, C_l, H/2^l, W/2^l)
        """
        features = []
        
        # F_0: embedding feature
        f0 = self.embed(x)
        features.append(f0)
        
        # F_1 through F_4
        current = f0
        for stage in self.stages:
            for block in stage:
                current = block(current)
            features.append(current)
        
        return features  # [F_0, F_1, F_2, F_3, F_4]