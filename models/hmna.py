import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiScaleContextDictionary(nn.Module):
    """Constructs compact context dictionary from multi-scale features."""
    
    def __init__(self, in_channels, out_channels, pooling_kernels=[2, 4, 8], 
                 dilation_rates=[1, 2, 4, 8]):
        super().__init__()
        
        self.pooling_kernels = pooling_kernels
        self.dilation_rates = dilation_rates
        
        # Pooling layers for context aggregation
        self.pools = nn.ModuleList()
        for k in pooling_kernels:
            self.pools.append(nn.MaxPool2d(k, k))
        
        # Dilated convolutions for context expansion
        self.dilated_convs = nn.ModuleList()
        for r in dilation_rates:
            self.dilated_convs.append(
                nn.Conv2d(in_channels, out_channels, 3, 1, r, r)
            )
        
        # Compression
        self.compress = nn.Conv2d(out_channels * len(dilation_rates), out_channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-scale pooling
        pooled_features = []
        for pool in self.pools:
            p = pool(x)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            pooled_features.append(p)
        
        # Concatenate pooled features
        pooled = torch.cat(pooled_features, dim=1)
        
        # Apply dilated convolutions
        dilated_features = []
        for conv in self.dilated_convs:
            d = conv(pooled)
            dilated_features.append(d)
        
        # Compress
        dictionary = self.compress(torch.cat(dilated_features, dim=1))
        return dictionary


class LocalRefinement(nn.Module):
    """Multi-receptive-field local refinement with channel groups."""
    
    def __init__(self, channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        
        self.kernels = kernels
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups
        
        self.branches = nn.ModuleList()
        for k in kernels:
            padding = k // 2
            branch = nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, k, 1, padding, groups=self.group_channels),
                nn.BatchNorm2d(self.group_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Learnable scale coefficients
        self.beta = nn.Parameter(torch.zeros(self.num_groups))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split into groups
        groups = x.chunk(self.num_groups, dim=1)
        
        # Process each group
        refined = []
        for i, group in enumerate(groups):
            r = self.branches[i](group) + group
            refined.append(r)
        
        # Weighted combination
        weights = torch.softmax(self.beta, dim=0)
        output = torch.zeros_like(x)
        
        for i, r in enumerate(refined):
            output += weights[i] * r
        
        return output


class HMNA(nn.Module):
    """
    Hierarchical Multi-Scale Non-Local Attention.
    Combines compact context dictionary with local multi-receptive-field refinement.
    """
    
    def __init__(self, channels, compressed_channels=64, 
                 pooling_kernels=[2, 4, 8], dilation_rates=[1, 2, 4, 8],
                 local_kernels=[3, 5, 7, 9]):
        super().__init__()
        
        self.compressed_channels = compressed_channels
        
        # Context dictionary construction
        self.dictionary = MultiScaleContextDictionary(
            channels, compressed_channels, pooling_kernels, dilation_rates
        )
        
        # Query, key, value projections
        self.q_proj = nn.Conv2d(channels, compressed_channels, 1)
        self.k_proj = nn.Conv2d(compressed_channels, compressed_channels, 1)
        self.v_proj = nn.Conv2d(compressed_channels, compressed_channels, 1)
        self.out_proj = nn.Conv2d(compressed_channels, channels, 1)
        
        # Local refinement
        self.local_refinement = LocalRefinement(channels, local_kernels)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            Refined feature map
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Build context dictionary
        dictionary = self.dictionary(x)  # (B, C_dict, H, W)
        dict_flat = dictionary.flatten(2).transpose(1, 2)  # (B, N, C_dict)
        
        # Query from original features
        query = self.q_proj(x).flatten(2).transpose(1, 2)  # (B, N, C_comp)
        
        # Key and value from dictionary
        key = self.k_proj(dictionary).flatten(2).transpose(1, 2)  # (B, N, C_comp)
        value = self.v_proj(dictionary).flatten(2).transpose(1, 2)  # (B, N, C_comp)
        
        # Non-local attention with compact dictionary
        scale = self.compressed_channels ** -0.5
        attn = torch.matmul(query, key.transpose(1, 2)) * scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        
        attended = torch.matmul(attn, value)  # (B, N, C_comp)
        attended = attended.transpose(1, 2).reshape(B, self.compressed_channels, H, W)
        
        # Output projection
        out = self.out_proj(attended)
        out = out + x  # Residual connection
        
        # Local refinement
        out = self.local_refinement(out)
        out = out + x  # Residual connection
        
        return out
