import os
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import SimpleITK as sitk


class ACDCDataset(Dataset):
    """ACDC dataset loader with scribble support."""
    
    def __init__(self, data_dir, split_file, scribble_dir=None, 
                 phase='all', target_size=256, augment=False):
        """
        Args:
            data_dir: Path to ACDC dataset
            split_file: Path to patient split file
            scribble_dir: Path to scribble annotations (if None, use dense masks)
            phase: 'ED', 'ES', or 'all'
            target_size: Target image size
            augment: Apply augmentation
        """
        self.data_dir = data_dir
        self.scribble_dir = scribble_dir
        self.target_size = target_size
        self.augment = augment
        
        # Load patient IDs
        with open(split_file, 'r') as f:
            self.patients = [line.strip() for line in f.readlines()]
        
        # Collect all slices
        self.samples = []
        for patient in self.patients:
            patient_dir = os.path.join(data_dir, patient)
            for phase in ['ED', 'ES']:
                if phase == 'all' or phase == phase:
                    img_path = os.path.join(patient_dir, f'{patient}_{phase}.nii.gz')
                    mask_path = os.path.join(patient_dir, f'{patient}_{phase}_gt.nii.gz')
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append({
                            'image': img_path,
                            'mask': mask_path,
                            'patient': patient,
                            'phase': phase
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        img = sitk.ReadImage(sample['image'])
        mask = sitk.ReadImage(sample['mask'])
        
        img = sitk.GetArrayFromImage(img)
        mask = sitk.GetArrayFromImage(mask)
        
        # Process each slice
        processed_slices = []
        for i in range(img.shape[0]):
            img_slice = img[i]
            mask_slice = mask[i]
            
            # Normalize
            img_slice = self._normalize(img_slice)
            
            # Resize
            img_slice, mask_slice = self._resize(img_slice, mask_slice)
            
            # Augmentation
            if self.augment:
                img_slice, mask_slice = self._augment(img_slice, mask_slice)
            
            # Convert to tensor
            img_slice = torch.FloatTensor(img_slice).unsqueeze(0)
            mask_slice = torch.LongTensor(mask_slice)
            
            processed_slices.append((img_slice, mask_slice))
        
        # Return all slices (stacked)
        images = torch.stack([s[0] for s in processed_slices])
        masks = torch.stack([s[1] for s in processed_slices])
        
        return images, masks
    
    def _normalize(self, img):
        """Z-score normalization."""
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img.astype(np.float32)
    
    def _resize(self, img, mask):
        """Resize to target size."""
        from skimage.transform import resize
        H, W = img.shape
        img = resize(img, (self.target_size, self.target_size), preserve_range=True)
        mask = resize(mask, (self.target_size, self.target_size), preserve_range=True, order=0)
        return img.astype(np.float32), mask.astype(np.int64)
    
    def _augment(self, img, mask):
        """Apply augmentations."""
        import random
        from scipy.ndimage import rotate, zoom
        from skimage.transform import warp, AffineTransform
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = rotate(img, angle, reshape=False, order=1)
            mask = rotate(mask, angle, reshape=False, order=0)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(0.85, 1.15)
            img = zoom(img, (scale, scale), order=1)
            mask = zoom(mask, (scale, scale), order=0)
            # Crop to original size
            H, W = img.shape
            if scale > 1:
                img = img[(H - self.target_size)//2:(H + self.target_size)//2,
                         (W - self.target_size)//2:(W + self.target_size)//2]
                mask = mask[(H - self.target_size)//2:(H + self.target_size)//2,
                           (W - self.target_size)//2:(W + self.target_size)//2]
            else:
                img = np.pad(img, ((H//2, H//2), (W//2, W//2)), mode='constant')
                mask = np.pad(mask, ((H//2, H//2), (W//2, W//2)), mode='constant')
                img = img[:self.target_size, :self.target_size]
                mask = mask[:self.target_size, :self.target_size]
        
        # Horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        
        return img, mask