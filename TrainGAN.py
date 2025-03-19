import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from model import Generator, Discriminator  # Ensure you have the model.py file with Generator and Discriminator classes

# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 50
noise_dim = 100
image_size = 28
image_channels = 1
save_interval = 10  # Save images every 10 epochs

# Create folder to save generated images
os.makedirs('generated_images', exist_ok=True)

# Data Loading (MNIST dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Initialize Generator and Discriminator
generator = Generator(noise_dim, image_channels, image_size)
discriminator = Discriminator(image_channels, image_size)

# Optimizers
optimizer_G = Adam(generator.parameters(), lr=lr)
optimizer_D = Adam(discriminator.parameters(), lr=lr)

# Loss Function (Binary Cross-Entropy)
criterion = nn.BCELoss()

# Function to save generated images
def save_generated_images(epoch, gen_imgs):
    gen_imgs = (gen_imgs + 1) / 2  # Rescale images to [0, 1]
    gen_imgs = gen_imgs.view(-1, 1, image_size, image_size)  # Reshape to [batch_size, channels, height, width]
    for i in range(gen_imgs.size(0)):
        img = gen_imgs[i].detach().squeeze().cpu().numpy()  # Detach from computation graph
        img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
        img = Image.fromarray(img, mode='L')  # Convert to grayscale PIL image
        img.save(f"generated_images/epoch_{epoch}_img_{i}.png")  # Save image

# Training Loop
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # Create labels for real and fake images
        valid = torch.ones(imgs.size(0), 1)  # Real images
        fake = torch.zeros(imgs.size(0), 1)  # Fake images

        # Train the Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from fake images
        real_loss = criterion(discriminator(imgs), valid)
        z = torch.randn(imgs.size(0), noise_dim)  # Generate random noise
        gen_imgs = generator(z)  # Generate fake images
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)  # Detach to not backprop through generator

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train the Generator
        optimizer_G.zero_grad()

        # Generator tries to fool the discriminator
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Print log info
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Save generated images every few epochs
    if epoch % save_interval == 0:
        save_generated_images(epoch, gen_imgs)

print("Training finished.")
