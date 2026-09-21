import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class CNNExpert(nn.Module):
    """
    CNN decoder expert for local texture and boundary evidence.
    U-Net-style decoder with skip connections.
    """
    
    def __init__(self, decoder_channels=[256, 128, 64, 32], 
                 skip_channels=[64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.decoder_channels = decoder_channels
        
        # Decoder stages
        self.decoder_blocks = nn.ModuleList()
        self.skip_project = nn.ModuleList()
        
        in_ch = skip_channels[-1]  # F_4
        
        for i, out_ch in enumerate(decoder_channels):
            # Skip projection
            skip_idx = 4 - i - 1
            if skip_idx >= 0:
                skip_proj = nn.Conv2d(skip_channels[skip_idx], out_ch, 1)
                self.skip_project.append(skip_proj)
            else:
                self.skip_project.append(nn.Identity())
            
            # Decoder block
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvBlock(in_ch, out_ch),
                ConvBlock(out_ch, out_ch)
            )
            self.decoder_blocks.append(block)
            in_ch = out_ch
        
        # Final output
        self.output = nn.Conv2d(decoder_channels[-1], 3, 1)  # 3 classes
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List [F_0, F_1, F_2, F_3, F_4]
        Returns:
            Logits for 3 classes
        """
        x = encoder_features[-1]  # F_4
        
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            
            # Add skip connection
            if i < len(self.skip_project):
                skip = self.skip_project[i](encoder_features[4 - i - 1])
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # Final output
        logits = self.output(x)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return logits
