import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""
    
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and MLP."""
    
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerExpert(nn.Module):
    """
    Transformer decoder expert for global anatomical context.
    Uses windowed attention at fine resolutions and global attention at coarse resolutions.
    """
    
    def __init__(self, depths=[2, 2, 6, 2], num_heads=8, 
                 embed_dims=[96, 192, 384, 768], window_size=8,
                 skip_channels=[64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.depths = depths
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.window_size = window_size
        
        # Project encoder features to transformer dimensions
        self.skip_project = nn.ModuleList()
        for i, ch in enumerate(skip_channels[:4]):
            self.skip_project.append(
                nn.Conv2d(ch, embed_dims[i], 1) if i < len(embed_dims) else nn.Identity()
            )
        
        # Decoder stages (from coarse to fine)
        self.decoder_stages = nn.ModuleList()
        in_dim = embed_dims[-1]
        
        for i in range(len(embed_dims) - 1, -1, -1):
            stage = nn.ModuleList()
            dim = embed_dims[i]
            
            # Upsample
            if i < len(embed_dims) - 1:
                stage.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_dim, dim, 1)
                ))
            
            # Transformer blocks
            for _ in range(depths[i]):
                stage.append(
                    SwinTransformerBlock(dim, num_heads, min(window_size, 2**i))
                )
            
            self.decoder_stages.append(stage)
            in_dim = dim
        
        self.decoder_stages = self.decoder_stages[::-1]
        
        # Final projection to logits
        self.output_proj = nn.Conv2d(embed_dims[0], 3, 1)  # 3 classes
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List [F_0, F_1, F_2, F_3, F_4]
        Returns:
            Logits for 3 classes
        """
        # Start from deepest feature
        x = encoder_features[-1]  # F_4
        
        # Project to first transformer dimension
        x = self.skip_project[-1](x)
        
        # Decode through stages
        for stage_idx, stage in enumerate(self.decoder_stages):
            # Upsample if not first stage
            if stage_idx > 0 and len(stage) > 0:
                if isinstance(stage[0], nn.Sequential):
                    x = stage[0](x)
                    transformer_blocks = stage[1:]
                else:
                    transformer_blocks = stage
            else:
                transformer_blocks = stage
            
            # Add skip connection
            skip_idx = 4 - stage_idx - 1
            if skip_idx >= 0:
                skip = self.skip_project[skip_idx](encoder_features[skip_idx])
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
            
            # Apply transformer blocks
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            for block in transformer_blocks:
                x = block(x)
            
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Final projection
        logits = self.output_proj(x)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return logits
