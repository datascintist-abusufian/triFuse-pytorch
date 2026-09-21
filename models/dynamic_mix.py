import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMix(nn.Module):
    """
    Dynamic Mix gating network.
    Learns pixel-wise expert weights and performs confidence-filtered fusion.
    """
    
    def __init__(self, num_classes=3, hidden_dim=64):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Gating network: takes concatenated expert logits
        # Input: (B, 3 * num_classes, H, W)
        self.gate = nn.Sequential(
            nn.Conv2d(3 * num_classes, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1)  # 3 experts
        )
    
    def forward(self, logits_T, logits_C, logits_S, threshold=0.8, 
                pseudo_labels=None, label_mask=None):
        """
        Args:
            logits_T: Transformer expert logits (B, K, H, W)
            logits_C: CNN expert logits (B, K, H, W)
            logits_S: SR expert logits (B, K, H, W)
            threshold: Confidence threshold for pseudo-label generation
            pseudo_labels: Ground truth labels for labeled pixels (or None)
            label_mask: Binary mask indicating labeled pixels
        
        Returns:
            fused_logits: Fused logits
            pseudo_labels: Generated pseudo-labels for unlabeled pixels
        """
        # Concatenate logits
        concat = torch.cat([logits_T, logits_C, logits_S], dim=1)  # (B, 3*K, H, W)
        
        # Predict gating weights
        gate_logits = self.gate(concat)  # (B, 3, H, W)
        weights = F.softmax(gate_logits, dim=1)  # (B, 3, H, W)
        
        # Fuse logits
        fused_logits = weights[:, 0:1] * logits_T + \
                       weights[:, 1:2] * logits_C + \
                       weights[:, 2:3] * logits_S  # (B, K, H, W)
        
        # Generate pseudo-labels
        if pseudo_labels is not None and label_mask is not None:
            # Get probabilities
            probs = F.softmax(fused_logits, dim=1)  # (B, K, H, W)
            max_probs, preds = probs.max(dim=1)  # (B, H, W)
            
            # Initialize with ground truth for labeled pixels
            pseudo = pseudo_labels.clone()
            pseudo[~label_mask] = -1  # Mark unlabeled
            
            # For unlabeled pixels with confidence >= threshold
            unlabeled_mask = ~label_mask
            confident_mask = (max_probs >= threshold) & unlabeled_mask
            pseudo[confident_mask] = preds[confident_mask]
            
            return fused_logits, pseudo, weights
        else:
            return fused_logits, None, weights
