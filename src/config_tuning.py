# src/config_tuning.py

class Config:
    # Model parameters
    latent_dim = 128
    input_shape = (1, 224, 224)  # example for grayscale mammograms, adjust as needed

    # Training parameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50

    # Diffusion or VAE-specific parameters
    diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # Add paths for saving models and logs
    model_save_path = "./models/"
    log_dir = "./logs/"

# Optionally, instantiate a global configuration object
config = Config()
