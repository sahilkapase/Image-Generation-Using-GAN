# Image-Generation-Using-GAN

Overview

This project implements a Generative Adversarial Network (GAN) to generate realistic human faces. The GAN is trained using a human face dataset (e.g., CelebA, FFHQ) and progressively improves the quality of generated images through adversarial training.

Dataset

We use the CelebA (CelebFaces Attributes Dataset), which contains over 200,000 celebrity images. The dataset is available at:

CelebA Dataset

FFHQ Dataset (Flickr-Faces-HQ)

Model Architecture

This project employs a standard DCGAN (Deep Convolutional GAN) architecture, consisting of:

Generator:

Transposed Convolution layers to upsample images.

Batch Normalization to stabilize training.

Leaky ReLU activation functions.

Discriminator:

Convolutional layers for feature extraction.

Leaky ReLU for better gradient flow.

Sigmoid activation to output real/fake probability.

Installation

To set up the project, install the required dependencies:

pip install torch torchvision matplotlib numpy pillow tqdm
pip install tensorflow-datasets  # If using TF datasets

Training

Run the following script to train the GAN model:

python train.py --epochs 100 --batch_size 64 --dataset celebA

Training Parameters:

Epochs: 30

Batch Size: 64 (default)

Dataset: CelebA

Inference

To generate new images using a trained model, run:

python generate.py --model_path checkpoints/generator.pth

Results

Below are some generated images after training for 30 epochs:
![image](https://github.com/user-attachments/assets/d8d7c2d6-0353-41ae-9137-7aaf13d6ec30)
