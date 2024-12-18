import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class WeldingDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        json_path = os.path.join(self.json_dir, image_name.replace('.png', '.json'))

        # 이미지 불러오기
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        
        # JSON 파일에서 용접선 개수 읽기
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            vertices_count = sum(len(line["vertices"]) for line in json_data["lines"])
            if vertices_count == 8:
                label = 2  # 용접선 2개
            elif vertices_count == 4:
                label = 1  # 용접선 1개
            else:
                label = 0  # 용접선 0개

        return image, label
