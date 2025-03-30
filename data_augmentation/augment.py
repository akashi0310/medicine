import os
import cv2
import albumentations as A
from tqdm import tqdm

SCAN_DIR = "/kaggle/input/lung2data/train_2/origin_2"
MASK_DIR = "/kaggle/input/lung2data/train_2/mask_2"
AUGMENTED_SCAN_DIR = "augmented_scans"
AUGMENTED_MASK_DIR = "augmented_masks"
os.makedirs(AUGMENTED_SCAN_DIR, exist_ok=True)
os.makedirs(AUGMENTED_MASK_DIR, exist_ok=True)

transform = A.Compose([A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(alpha=50, sigma=5, alpha_affine=10, p=0.5),
    A.Resize(height=256, width=256),
], additional_targets={'mask': 'mask'})
scan_files = sorted(os.listdir(SCAN_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

for scan_file, mask_file in tqdm(zip(scan_files, mask_files), total=len(scan_files)):
    scan_path = os.path.join(SCAN_DIR, scan_file)
    mask_path = os.path.join(MASK_DIR, mask_file)
    ct_scan = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    for i in range(3):
        augmented = transform(image=ct_scan, mask=mask)
        aug_scan = augmented['image']
        aug_mask = augmented['mask']
        base_name = scan_file.split('.')[0]
        suffix = f"_{i:02d}.png"
        cv2.imwrite(os.path.join(AUGMENTED_SCAN_DIR, f"{base_name}{suffix}"), aug_scan)
        cv2.imwrite(os.path.join(AUGMENTED_MASK_DIR, f"{base_name}_mask{suffix}"), aug_mask)

print("Augmentation completed successfully!")