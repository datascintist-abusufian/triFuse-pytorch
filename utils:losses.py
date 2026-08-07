import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss."""
    
    def __init__(self, num_classes=3, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Logits (B, K, H, W)
            target: Labels (B, H, W)
            mask: Valid pixels mask (B, H, W)
        """
        B, K, H, W = pred.shape
        pred = F.softmax(pred, dim=1)
        
        if mask is not None:
            pred = pred * mask.unsqueeze(1)
            target = target * mask
        
        dice_loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c].reshape(B, -1)
            target_c = (target == c).float().reshape(B, -1)
            
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice).mean()
        
        return dice_loss / self.num_classes


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with masking for valid pixels."""
    
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Logits (B, K, H, W)
            target: Labels (B, H, W) with ignore_index for invalid pixels
        """
        return F.cross_entropy(pred, target, ignore_index=self.ignore_index)


class CombinedLoss(nn.Module):
    """Combined loss: CrossEntropy + Dice."""
    
    def __init__(self, num_classes=3, dice_weight=0.5, ce_weight=0.5, ignore_index=-1):
        super().__init__()
        self.ce_loss = MaskedCrossEntropyLoss(ignore_index)
        self.dice_loss = DiceLoss(num_classes)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Logits (B, K, H, W)
            target: Labels (B, H, W) with ignore_index for invalid pixels
            mask: Binary mask for valid pixels (optional)
        """
        if mask is not None:
            target = target.clone()
            target[~mask] = -1  # Ignore invalid pixels
        
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target, mask)
        
        return self.ce_weight * ce + self.dice_weight * dice


def compute_loss(outputs, targets, label_mask, lambda_aux=0.3, lambda_rev=0.2):
    """
    Compute total loss for TriFuse-SRNet.
    
    Args:
        outputs: Dict from model forward pass
        targets: Ground truth labels (B, H, W)
        label_mask: Binary mask for labeled pixels (B, H, W)
        lambda_aux: Weight for auxiliary losses
        lambda_rev: Weight for reverse supervision
    
    Returns:
        Total loss and individual components
    """
    loss_fn = CombinedLoss(num_classes=3)
    
    # Main fusion loss (only on labeled pixels)
    loss_main = loss_fn(outputs['fused'], targets, label_mask)
    
    # Auxiliary losses (only on labeled pixels)
    loss_T = loss_fn(outputs['logits_T'], targets, label_mask)
    loss_C = loss_fn(outputs['logits_C'], targets, label_mask)
    loss_S = loss_fn(outputs['logits_S'], targets, label_mask)
    loss_aux = (loss_T + loss_C + loss_S) / 3
    
    # Reverse supervision (on confident unlabeled pixels)
    if outputs['pseudo_labels'] is not None:
        # Create mask for confident unlabeled pixels
        confident_mask = (outputs['pseudo_labels'] != -1) & (~label_mask)
        pseudo_targets = outputs['pseudo_labels'].clone()
        pseudo_targets[~confident_mask] = -1  # Ignore invalid
        
        # Apply to all experts
        loss_rev_T = loss_fn(outputs['logits_T'], pseudo_targets)
        loss_rev_C = loss_fn(outputs['logits_C'], pseudo_targets)
        loss_rev_S = loss_fn(outputs['logits_S'], pseudo_targets)
        loss_rev = (loss_rev_T + loss_rev_C + loss_rev_S) / 3
    else:
        loss_rev = torch.tensor(0.0, device=targets.device)
    
    # Total loss
    total_loss = loss_main + lambda_aux * loss_aux + lambda_rev * loss_rev
    
    return {
        'total': total_loss,
        'main': loss_main,
        'aux': loss_aux,
        'rev': loss_rev
    }