# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDiffusionModel(nn.Module):
    def __init__(self, config):
        super(LatentDiffusionModel, self).__init__()
        self.latent_dim = config.latent_dim
        
        # Example encoder architecture: simple convolutional network
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (config.input_shape[1] // 4) * (config.input_shape[2] // 4), config.latent_dim)
        )
        
        # Example decoder: similarly simplistic
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 64 * (config.input_shape[1] // 4) * (config.input_shape[2] // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, config.input_shape[1] // 4, config.input_shape[2] // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode image to latent space
        latent = self.encoder(x)
        # Decode latent space back to image
        recon = self.decoder(latent)
        return recon, latent

if __name__ == "__main__":
    # Simple test: create a model and see the output shapes
    from config_tuning import config
    model = LatentDiffusionModel(config)
    dummy_input = torch.randn(4, *config.input_shape)
    recon, latent = model(dummy_input)
    print("Reconstructed image shape:", recon.shape)
    print("Latent representation shape:", latent.shape)
