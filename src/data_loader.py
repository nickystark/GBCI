import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

class MammogramDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_masks=True, use_metadata=True):
        self.data_dir = data_dir
        self.transform = transform
        self.use_masks = use_masks
        self.use_metadata = use_metadata

        # Carica il CSV con le informazioni sulle immagini
        self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        # Mappature per codificare attributi testuali in vettori numerici
        self.density_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.view_map = {"CC": 0, "MLO": 1}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(img_path).convert("L")  # grayscale

        # Carica la maschera se richiesta
        mask = None
        if self.use_masks:
            mask_path = os.path.join(self.mask_dir, f"{image_id}_mask.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", image.size)  # maschera vuota

        # Applica le trasformazioni
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)

        # Crea l'immagine mascherata (masked image = image * mask)
        if mask is not None:
            masked_image = image * (mask > 0.5).float()

        # Prepara attributi anatomici (esempio: densità, vista)
        attributes = torch.tensor([])
        if self.use_metadata:
            # Codifica densità mammaria (A,B,C,D) come one-hot
            density = torch.zeros(4)
            density_idx = self.density_map.get(str(row.get("density", "A")), 0)
            density[density_idx] = 1.0
            # Codifica vista (CC/MLO) come one-hot
            view = torch.zeros(2)
            view_idx = self.view_map.get(str(row.get("view", "CC")), 0)
            view[view_idx] = 1.0
            # Puoi aggiungere altri attributi qui (es. tipo lesione)
            attributes = torch.cat([density, view])

        # Restituisci immagine, maschera e attributi
        return {
            "image": image,
            "mask": mask if mask is not None else torch.zeros_like(image),
            "masked_image": masked_image if mask is not None else torch.zeros_like(image),  # immagine mascherata per il condizionamento
            "attributes": attributes
        }

def get_data_loader(data_dir, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = MammogramDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Esempio di utilizzo:
# train_loader = get_data_loader("./data/processed", batch_size=32)