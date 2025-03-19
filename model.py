import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import imageio.v2 as imageio  # Use imageio.v2 to suppress warnings

# Custom Dataset for loading images without subdirectories
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

        if self.transform:
            image = self.transform(image)

        return image

# Define the GAN model
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 128 * 128),  # Assuming output images are 128x128 RGB
            nn.Tanh()  # Output is in the range [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 128, 128)  # Reshape to image format
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 128 * 128, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output is in the range [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten the image
        validity = self.model(img_flat)
        return validity

# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 50
noise_dim = 100
celeba_root = r'D:\preprocessed_images'  # Update this path
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Load the dataset
celeba_dataset = CustomImageDataset(root_dir=celeba_root, transform=transform)
data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(noise_dim)
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss Function
criterion = nn.BCELoss()

# Function to save generated images
def save_generated_images(epoch, gen_imgs):
    gen_imgs = (gen_imgs + 1) / 2  # Rescale images to [0, 1]
    gen_imgs = gen_imgs.view(-1, 3, 128, 128)  # Reshape to [batch_size, channels, height, width]
    for i in range(gen_imgs.size(0)):
        img = gen_imgs[i].detach().squeeze().cpu().numpy()  # Detach from computation graph
        img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
        img = Image.fromarray(img.transpose(1, 2, 0), 'RGB')  # Convert to RGB PIL image
        img.save(f"generated_images/epoch_{epoch}_img_{i}.png")  # Save image

# Create directory to save generated images
os.makedirs('generated_images', exist_ok=True)

# Training Loop
for epoch in range(num_epochs):
    for i, imgs in enumerate(data_loader):
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
    if epoch % 10 == 0:  # Adjust the save interval as needed
        save_generated_images(epoch, gen_imgs)

print("Training finished.")

# Function to convert generated images to GIF
def images_to_gif(image_folder, gif_name):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))

    if not images:
        raise ValueError("No images found in the folder.")
    imageio.mimsave(gif_name, images, fps=5)

# Specify output GIF file
gif_name = r'D:\SYCSAIML\output_funny_gif.gif'  # Save the GIF in a writable directory
images_to_gif('generated_images', gif_name)  # Convert generated images to GIF
