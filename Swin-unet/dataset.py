import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class Synapse_dataset(Dataset):
    def __init__(self, data_root, label_root, transform=None, output_size=(224, 224)):
        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform
        self.output_size = output_size
        self.img_extensions = ('.png',)
        
        self.image_paths = []
        self.label_paths = []
        for f in os.listdir(data_root):
            if f.lower().endswith(self.img_extensions):
                self.image_paths.append(os.path.join(data_root, f))
        for f in os.listdir(label_root):
            if f.lower().endswith(self.img_extensions):
                self.label_paths.append(os.path.join(label_root, f))
        self.image_paths.sort()
        self.label_paths.sort()
        
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels not equal"

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = Image.open(self.label_paths[idx]).convert('L')
        if (image.width, image.height) != self.output_size:
            image = image.resize(self.output_size, Image.BICUBIC)
            label = label.resize(self.output_size, Image.NEAREST)
        image_array = np.array(image, dtype=np.float32) / 255.0
        label_array = (np.array(label) > 0).astype(np.int64)
        sample = {'image': image_array, 'label': label_array}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = {
                'image': torch.from_numpy(image_array).unsqueeze(0),
                'label': torch.from_numpy(label_array)
            }
        return sample