# GBCI: Generative Breast Cancer Imaging

## Overview
Generative models are transforming medical imaging by enabling **simulation, augmentation, and in-silico patient profiling**. This project focuses on developing a **latent diffusion model (LDM) and/or a VAE-based pipeline** trained on mammograms (e.g., VinDr-Mammo or INbreast) to generate **synthetic cancer-positive and -negative mammograms**. The goal is to explore **semantic conditioning**, ensuring clinically relevant attributes are captured, such as **mass density, calcifications, and tissue asymmetries**.

## Project Goals
- Implement a **diffusion-based generative model** for mammogram synthesis.
- Explore **latent-space representations** that capture cancer-relevant features.
- **Condition** image generation using metadata such as BI-RADS scores, lesion type, or patient profiles.
- Compare **VAE-based approaches** and evaluate generated samples for diagnostic reliability.

## Team Contributions
- **Feature Conditioning:** Cross-attention or FiLM layers to incorporate metadata and radiology reports.
- **Reconstruction Evaluation:** Testing synthetic images through classification/segmentation models for diagnostic robustness.
- **Domain Translation:** Generating lesion-containing images from healthy scans to simulate **disease progression**.

## Honors Extensions
For advanced implementation:
- **Longitudinal Modeling:** Simulating lesion evolution using **flow-matching** or **temporal diffusion models**.
- **Counterfactual Synthesis:** Generating alternative scenarios based on real patient data.
- **Multimodal Generation:** Producing coherent mammogram-biopsy image pairs.

## Repository Structure
