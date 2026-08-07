import torch
import numpy as np
from scipy.spatial import distance


def dice_score(pred, target, num_classes=3, smooth=1e-6):
    """Compute Dice score per class."""
    pred = pred.argmax(dim=1) if pred.dim() > 2 else pred
    dice = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dice.append(0.0)
        else:
            dice.append((2 * intersection + smooth) / (union + smooth))
    
    return np.array(dice)


def volume_error(pred, target, num_classes=3, spacing=1.5):
    """Compute relative volume error per class."""
    pred = pred.argmax(dim=1) if pred.dim() > 2 else pred
    volume_errors = []
    
    for c in range(num_classes):
        pred_vol = (pred == c).sum().item()
        target_vol = (target == c).sum().item()
        if target_vol == 0:
            volume_errors.append(0.0)
        else:
            volume_errors.append(abs(pred_vol - target_vol) / target_vol)
    
    return np.array(volume_errors)


def hd95(pred, target, num_classes=3, spacing=1.5):
    """Compute 95th percentile Hausdorff distance."""
    pred = pred.argmax(dim=1) if pred.dim() > 2 else pred
    hd_values = []
    
    for c in range(num_classes):
        pred_c = (pred == c).cpu().numpy()
        target_c = (target == c).cpu().numpy()
        
        if pred_c.sum() == 0 or target_c.sum() == 0:
            hd_values.append(0.0)
            continue
        
        # Get coordinates
        pred_coords = np.argwhere(pred_c)
        target_coords = np.argwhere(target_c)
        
        # Compute distances
        distances = distance.cdist(pred_coords, target_coords)
        min_dist = distances.min(axis=1)
        
        # 95th percentile
        hd = np.percentile(min_dist, 95) * spacing
        hd_values.append(hd)
    
    return np.array(hd_values)


def assd(pred, target, num_classes=3, spacing=1.5):
    """Compute Average Symmetric Surface Distance."""
    pred = pred.argmax(dim=1) if pred.dim() > 2 else pred
    assd_values = []
    
    for c in range(num_classes):
        pred_c = (pred == c).cpu().numpy()
        target_c = (target == c).cpu().numpy()
        
        if pred_c.sum() == 0 or target_c.sum() == 0:
            assd_values.append(0.0)
            continue
        
        # Get surface coordinates
        from scipy.ndimage import binary_erosion, binary_dilation
        pred_surface = pred_c ^ binary_erosion(pred_c)
        target_surface = target_c ^ binary_erosion(target_c)
        
        pred_coords = np.argwhere(pred_surface)
        target_coords = np.argwhere(target_surface)
        
        # Compute distances
        distances_pred = distance.cdist(pred_coords, target_coords)
        distances_target = distance.cdist(target_coords, pred_coords)
        
        # Average symmetric distances
        d_pred = distances_pred.min(axis=1).mean()
        d_target = distances_target.min(axis=1).mean()
        
        assd = (d_pred + d_target) / 2 * spacing
        assd_values.append(assd)
    
    return np.array(assd_values)


def compute_all_metrics(pred, target, num_classes=3, spacing=1.5):
    """Compute all metrics."""
    dice = dice_score(pred, target, num_classes)
    vol_error = volume_error(pred, target, num_classes, spacing)
    hd = hd95(pred, target, num_classes, spacing)
    as_sd = assd(pred, target, num_classes, spacing)
    
    return {
        'dice': dice,
        'volume_error': vol_error,
        'hd95': hd,
        'assd': as_sd
    }