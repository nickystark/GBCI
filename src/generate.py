# src/predict.py

import torch
from config_tuning import config
from model import LatentDiffusionModel
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path):
    model = LatentDiffusionModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_image(model, input_image):
    # For example, you might pass an image through the encoder and then decode.
    # This is a simple demonstration.
    with torch.no_grad():
        recon, _ = model(input_image)
    return recon

if __name__ == "__main__":
    # Example usage:
    model = load_model("./models/model_epoch50.pth")
    
    preprocess = transforms.Compose([
        transforms.Resize((config.input_shape[1], config.input_shape[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load an example image
    img = Image.open("./data/sample_mammogram.jpg").convert("L")
    input_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
    
    output_tensor = generate_image(model, input_tensor)
    
    # Convert tensor to image and save/display
    out_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    out_image.save("generated_sample.jpg")
    print("Image generated and saved as generated_sample.jpg")
