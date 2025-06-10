# src/data_loader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class MammogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform if transform is not None else transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale if needed
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(data_dir, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # adjust based on your dataset
    ])
    
    dataset = MammogramDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# Example usage:
# train_loader = get_data_loader("./data/processed", batch_size=32)
