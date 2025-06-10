# src/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from config_tuning import config
from data_loader import get_data_loader
from model import LatentDiffusionModel
import os

def train():
    # Initialize data loader
    train_loader = get_data_loader(data_dir="./data/processed", batch_size=config.batch_size)
    
    # Initialize model, loss and optimizer
    model = LatentDiffusionModel(config)
    model.train()
    criterion = nn.MSELoss()  # For reconstruction; adjust based on your objective
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for images in train_loader:
            optimizer.zero_grad()
            recon, _ = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoints periodically
        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)
        torch.save(model.state_dict(), os.path.join(config.model_save_path, f"model_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    train()
