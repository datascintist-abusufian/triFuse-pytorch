import os
import sys
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm
from glob import glob


def preprocess_acdc(input_dir, output_dir, target_size=256):
    """Preprocess ACDC dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    patient_dirs = glob(os.path.join(input_dir, 'patient*'))
    
    for patient_dir in tqdm(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        output_patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(output_patient_dir, exist_ok=True)
        
        # Find all NIfTI files
        for phase in ['ED', 'ES']:
            img_path = os.path.join(patient_dir, f'{patient_id}_{phase}.nii.gz')
            mask_path = os.path.join(patient_dir, f'{patient_id}_{phase}_gt.nii.gz')
            
            if not os.path.exists(img_path):
                continue
            
            # Load image
            img = nib.load(img_path)
            data = img.get_fdata()
            
            # Process each slice
            processed_slices = []
            for i in range(data.shape[2]):
                slice_data = data[:, :, i]
                slice_data = (slice_data - slice_data.mean()) / (slice_data.std() + 1e-8)
                slice_data = resize(slice_data, (target_size, target_size), preserve_range=True)
                processed_slices.append(slice_data.astype(np.float32))
            
            processed_img = np.stack(processed_slices, axis=2)
            
            # Save preprocessed image
            output_img = nib.Nifti1Image(processed_img, img.affine, img.header)
            nib.save(output_img, os.path.join(output_patient_dir, f'{patient_id}_{phase}.nii.gz'))
            
            # Process mask
            if os.path.exists(mask_path):
                mask = nib.load(mask_path)
                mask_data = mask.get_fdata()
                processed_mask = []
                for i in range(mask_data.shape[2]):
                    mask_slice = mask_data[:, :, i]
                    mask_slice = resize(mask_slice, (target_size, target_size), preserve_range=True, order=0)
                    processed_mask.append(mask_slice.astype(np.int64))
                
                processed_mask = np.stack(processed_mask, axis=2)
                output_mask = nib.Nifti1Image(processed_mask, mask.affine, mask.header)
                nib.save(output_mask, os.path.join(output_patient_dir, f'{patient_id}_{phase}_gt.nii.gz'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mscmrseg'])
    parser.add_argument('--target_size', type=int, default=256)
    args = parser.parse_args()
    
    if args.dataset == 'acdc':
        preprocess_acdc(args.input, args.output, args.target_size)
    else:
        print('MSCMRseg preprocessing not implemented in this example')
        # Add MSCMRseg preprocessing here


if __name__ == '__main__':
    main()
