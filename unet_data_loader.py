import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class WeldingDataset(Dataset):
    def __init__(self, data1_dir, data2_dir, data3_dir, transform, image_real_paths, mask_welding_paths, unprocessed_mask_paths):
        self.data1_dir = data1_dir # 이미지 저장폴더(data1)
        self.data2_dir = data2_dir # 이미지 저장폴더(data2)
        self.data3_dir = data3_dir # 이미지 저장폴더(data3)
        self.transform = transform 
        self.image_paths = image_real_paths 
        self.mask_paths = mask_welding_paths
        self.unprocessed_paths = unprocessed_mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data1_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        weld_mask_path = os.path.join(self.data2_dir, self.mask_paths[idx])
        weld_mask = Image.open(weld_mask_path).convert("L")

        unprocessed_mask_path = os.path.join(self.data3_dir, self.unprocessed_paths[idx])
        unprocessed_mask = Image.open(unprocessed_mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            weld_mask = self.transform(weld_mask)
            unprocessed_mask = self.transform(unprocessed_mask)

        return image, weld_mask, unprocessed_mask