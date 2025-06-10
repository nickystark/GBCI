
# src/metrics.py

import math
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
# If you decide to implement FID, consider importing a pre-trained model from torchvision and other utilities.

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
        img1 (np.ndarray): First image (as a numpy array).
        img2 (np.ndarray): Second image (as a numpy array).
    
    Returns:
        float: SSIM value between the two images.
    """
    # Ensure images have the same shape and proper data range
    ssim_value, _ = compare_ssim(img1, img2, full=True, data_range=img2.max() - img2.min())
    return ssim_value

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
        img1 (np.ndarray): First image (as a numpy array).
        img2 (np.ndarray): Second image (as a numpy array).
    
    Returns:
        float: PSNR value in decibels.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # Assuming the images are normalized between 0 and 1
    PIXEL_MAX = 1.0  
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def compute_fid(fake_images, real_images):
    """
    Compute the Fréchet Inception Distance (FID) between a set of generated (fake) images
    and real images. This is just a placeholder; actual FID calculation requires:
      - A pre-trained InceptionV3 model or similar to extract features.
      - Computation of the mean and covariance of features for both sets.
      - Calculation of the Fréchet distance.
    
    Parameters: 
        fake_images: Array or tensor of generated images.
        real_images: Array or tensor of real images.
        
    Returns:
        float: FID score.
    
    Note: You might consider using libraries like `pytorch-fid` or writing a dedicated FID routine.
    """
    raise NotImplementedError("FID computation needs a feature extractor and further implementation.")

if __name__ == "__main__":
    # Example usage: Creating dummy images to demonstrate SSIM and PSNR.
    img1 = np.random.rand(224, 224)  # Dummy image with values between 0 and 1
    img2 = np.random.rand(224, 224)  # Another dummy image

    ssim_value = compute_ssim(img1, img2)
    psnr_value = compute_psnr(img1, img2)
    
    print(f"SSIM: {ssim_value:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    # FID is not computed in this example.
