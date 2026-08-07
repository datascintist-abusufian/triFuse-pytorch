import torch
import torch.nn as nn
from .encoder import SharedEncoder
from .transformer_expert import TransformerExpert
from .cnn_expert import CNNExpert
from .sr_branch import SRBranch
from .hmna import HMNA
from .dynamic_mix import DynamicMix


class TriFuseSRNet(nn.Module):
    """
    TriFuse-SRNet: Dynamic Multi-Expert Fusion with Structural Recovery.
    """
    
    def __init__(self, in_channels=1, num_classes=3, embed_dim=64,
                 encoder_depths=[2, 2, 3, 3], encoder_channels=[64, 128, 256, 512, 1024],
                 transformer_depths=[2, 2, 6, 2], transformer_heads=8,
                 transformer_dims=[96, 192, 384, 768], window_size=8,
                 cnn_decoder_channels=[256, 128, 64, 32],
                 sr_projected_width=256, sr_num_blocks=4, sr_dilation=[1, 2, 4, 8],
                 hmna_compressed=64, hmna_pooling=[2, 4, 8], 
                 hmna_dilation=[1, 2, 4, 8], hmna_local=[3, 5, 7, 9]):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared encoder
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=encoder_depths,
            channels=encoder_channels
        )
        
        # Three experts
        self.transformer_expert = TransformerExpert(
            depths=transformer_depths,
            num_heads=transformer_heads,
            embed_dims=transformer_dims,
            window_size=window_size,
            skip_channels=encoder_channels
        )
        
        self.cnn_expert = CNNExpert(
            decoder_channels=cnn_decoder_channels,
            skip_channels=encoder_channels
        )
        
        self.sr_branch = SRBranch(
            skip_channels=encoder_channels,
            projected_width=sr_projected_width,
            num_blocks=sr_num_blocks,
            dilation_schedule=sr_dilation
        )
        
        # HMNA for SR refinement (applied to SR logits)
        self.hmna = HMNA(
            channels=num_classes,  # Refine logits
            compressed_channels=hmna_compressed,
            pooling_kernels=hmna_pooling,
            dilation_rates=hmna_dilation,
            local_kernels=hmna_local
        )
        
        # Dynamic Mix fusion
        self.dynamic_mix = DynamicMix(num_classes=num_classes)
    
    def forward(self, x, pseudo_labels=None, label_mask=None, threshold=0.8):
        """
        Args:
            x: Input image (B, 1, H, W)
            pseudo_labels: Ground truth labels for labeled pixels (B, H, W)
            label_mask: Binary mask for labeled pixels (B, H, W)
            threshold: Confidence threshold for pseudo-label generation
        
        Returns:
            Dictionary containing logits from all experts and fused output
        """
        # Encode
        encoder_features = self.encoder(x)
        
        # Get logits from each expert
        logits_T = self.transformer_expert(encoder_features)
        logits_C = self.cnn_expert(encoder_features)
        logits_S_raw = self.sr_branch(encoder_features)
        
        # Refine SR logits with HMNA
        logits_S = self.hmna(logits_S_raw)
        
        # Dynamic Mix fusion
        fused_logits, pseudo, weights = self.dynamic_mix(
            logits_T, logits_C, logits_S,
            threshold=threshold,
            pseudo_labels=pseudo_labels,
            label_mask=label_mask
        )
        
        return {
            'logits_T': logits_T,
            'logits_C': logits_C,
            'logits_S': logits_S,
            'logits_S_raw': logits_S_raw,
            'fused': fused_logits,
            'pseudo_labels': pseudo,
            'weights': weights
        }