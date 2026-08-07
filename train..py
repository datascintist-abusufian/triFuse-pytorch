import os
import sys
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TriFuseSRNet
from datasets.acdc import ACDCDataset
from datasets.mscmrseg import MSCMRsegDataset
from utils.losses import compute_loss
from utils.metrics import compute_all_metrics


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name, data_dir, split_file, scribble_dir=None,
                target_size=256, augment=False):
    """Get dataset loader."""
    if dataset_name.lower() == 'acdc':
        return ACDCDataset(data_dir, split_file, scribble_dir, target_size, augment)
    elif dataset_name.lower() == 'mscmrseg':
        return MSCMRsegDataset(data_dir, split_file, scribble_dir, target_size, augment)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_epoch(model, loader, optimizer, scheduler, device, cfg, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    losses_components = {'main': 0, 'aux': 0, 'rev': 0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Get images and masks
        images, masks = batch
        
        # For simplicity, use first slice in batch (handle multi-slice properly in real code)
        images = images[:, 0].to(device)  # (B, 1, H, W)
        masks = masks[:, 0].to(device)  # (B, H, W)
        
        # Create label mask (pixels with valid labels)
        label_mask = masks != -1
        
        # Forward pass
        outputs = model(images, pseudo_labels=masks, 
                       label_mask=label_mask, 
                       threshold=cfg['training']['pseudo_threshold'])
        
        # Compute loss
        losses = compute_loss(
            outputs, masks, label_mask,
            lambda_aux=cfg['training']['lambda_aux'],
            lambda_rev=cfg['training']['lambda_rev']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Update statistics
        total_loss += losses['total'].item()
        losses_components['main'] += losses['main'].item()
        losses_components['aux'] += losses['aux'].item()
        losses_components['rev'] += losses['rev'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'main': losses['main'].item(),
            'aux': losses['aux'].item(),
            'rev': losses['rev'].item()
        })
    
    scheduler.step()
    
    return {
        'total': total_loss / len(loader),
        'main': losses_components['main'] / len(loader),
        'aux': losses_components['aux'] / len(loader),
        'rev': losses_components['rev'] / len(loader)
    }


def validate(model, loader, device, cfg):
    """Validate model."""
    model.eval()
    all_metrics = {'dice': [], 'volume_error': [], 'hd95': [], 'assd': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images, masks = batch
            images = images[:, 0].to(device)
            masks = masks[:, 0].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics = compute_all_metrics(
                outputs['fused'], masks, 
                num_classes=cfg['model']['num_classes'],
                spacing=1.5
            )
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key].mean())
    
    # Average metrics
    results = {}
    for key in all_metrics:
        results[key] = np.mean(all_metrics[key])
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mscmrseg'])
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize model
    model = TriFuseSRNet(
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        embed_dim=cfg['model']['embed_dim'],
        encoder_depths=cfg['model']['encoder']['depths'],
        encoder_channels=cfg['model']['encoder']['channels'],
        transformer_depths=cfg['model']['transformer_expert']['depths'],
        transformer_heads=cfg['model']['transformer_expert']['num_heads'],
        transformer_dims=cfg['model']['transformer_expert']['embed_dims'],
        window_size=cfg['model']['transformer_expert']['window_size'],
        cnn_decoder_channels=cfg['model']['cnn_expert']['decoder_channels'],
        sr_projected_width=cfg['model']['sr_branch']['projected_width'],
        sr_num_blocks=cfg['model']['sr_branch']['num_blocks'],
        sr_dilation=cfg['model']['sr_branch']['dilation_schedule'],
        hmna_compressed=cfg['model']['hmna']['compressed_channels'],
        hmna_pooling=cfg['model']['hmna']['pooling_kernels'],
        hmna_dilation=cfg['model']['hmna']['dilation_rates'],
        hmna_local=cfg['model']['hmna']['local_kernels']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    
    # Load datasets
    data_dir = f'data/{args.dataset}'
    train_dataset = get_dataset(
        args.dataset, data_dir,
        f'splits/{args.dataset}_train.txt',
        target_size=cfg['data']['input_size'],
        augment=True
    )
    val_dataset = get_dataset(
        args.dataset, data_dir,
        f'splits/{args.dataset}_val.txt',
        target_size=cfg['data']['input_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['num_epochs'] - cfg['training']['warmup_epochs']
    )
    
    # Resume training
    start_epoch = 0
    best_dice = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
    
    # Tensorboard
    writer = SummaryWriter(f'logs/{args.dataset}_{args.seed}')
    
    # Training loop
    for epoch in range(start_epoch, cfg['training']['num_epochs']):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, epoch
        )
        
        # Log training losses
        for key, value in train_losses.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        
        # Validate
        if (epoch + 1) % 5 == 0 or epoch == cfg['training']['num_epochs'] - 1:
            val_metrics = validate(model, val_loader, device, cfg)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save best checkpoint
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'metrics': val_metrics
                }, f'checkpoints/{args.dataset}_{args.seed}_best.pth')
                print(f'✓ Best model saved! Dice: {best_dice:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_dice': best_dice
            }, f'checkpoints/{args.dataset}_{args.seed}_epoch_{epoch+1}.pth')
    
    print(f'Training complete! Best Dice: {best_dice:.4f}')
    writer.close()


if __name__ == '__main__':
    main()