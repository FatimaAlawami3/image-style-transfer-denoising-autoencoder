# Image Style Transfer with Denoising Autoencoder

## Overview

This project presents a **deep learning–based image style transfer system** designed to improve stylization quality for **noisy and low-resolution images**. The system combines a **Denoising Autoencoder** for image preprocessing with a **VGG19-based Neural Style Transfer** pipeline to generate visually coherent and high-quality stylized images.

The work was completed as part of a university deep learning course project and focuses on improving robustness and visual fidelity in neural style transfer.

---

## Problem Statement

Traditional neural style transfer methods often degrade significantly when applied to noisy or low-quality images. Noise and visual artifacts can distort textures, reduce clarity, and negatively affect the stability of the stylization process.

This project addresses this issue by introducing a **two-stage pipeline**:

1. Image denoising and refinement using a convolutional autoencoder
2. Artistic style transfer using a pretrained VGG19 network

---

## Dataset

Three Kaggle datasets were combined to build the training corpus:

* Artistic Images for Neural Style Transfer
* Artistic Styles (30k images dataset)
* Images for Style Transfer

After removing duplicates, normalization, and preprocessing:

* Total unique images: ~130
* Data split: 80% training, 10% validation, 10% testing
* Data augmentation applied to increase robustness and generalization

---

## Model Architecture

### Stage 1: Denoising Autoencoder

* Convolutional encoder–decoder architecture
* Trained to reconstruct clean images from noisy inputs
* Gaussian noise added during training
* Achieved ~88% validation pixel accuracy and ~80% test accuracy

This stage improves input image quality and stabilizes downstream style transfer.

### Stage 2: Neural Style Transfer (VGG19)

* Pretrained VGG19 used as a fixed feature extractor
* Content features extracted from deeper layers
* Style features captured using Gram matrices from early layers
* Generated image optimized using a weighted combination of content and style loss

---

## Training Details

* Optimizer: Adam
* Autoencoder learning rate: 1e-4
* Autoencoder epochs: up to 100 (early stopping)
* Style transfer optimization steps: 1500
* Losses:

  * Mean Squared Error (autoencoder)
  * Content loss
  * Style loss

---

## Results and Analysis

* Denoising autoencoder significantly improved input image quality
* Stylized outputs showed better texture stability and reduced artifacts
* Remaining challenges include:

  * Loss of fine details due to downsampling
  * High computational cost of iterative style transfer

---

## Project Structure

```
├── notebooks/
│   └── DL_Code.ipynb
│
├── report/
│   └── Image_Style_Transfer.pdf
│
├── README.md
└── requirements.txt
```

---

## Tools and Technologies

* Python
* TensorFlow / Keras
* VGG19
* NumPy, OpenCV
* Matplotlib
* Jupyter Notebook

---

## Key Learning Outcomes

* Designing multi-stage deep learning pipelines
* Applying denoising autoencoders for image preprocessing
* Implementing neural style transfer using perceptual loss
* Understanding the trade-off between visual quality and computational cost

---

## Team

This project was completed as a **university group project**.

---

## Future Work

* Replace iterative optimization with feed-forward networks or GANs
* Apply super-resolution techniques to preserve fine details
* Extend the approach to video-to-video style transfer
