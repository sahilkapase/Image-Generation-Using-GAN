import imageio.v2 as imageio
import os

image_folder = r'D:\SYCSAIML\SEM@5\ML\MachineLearningCP\pythonProject\.venv\generated_images'
gif_name = r'D:\SYCSAIML\output_funny_gif.gif'


def images_to_gif(image_folder, gif_name):
    images = []
    print(f"Files in {image_folder}: {os.listdir(image_folder)}")  # Debug: List files in folder
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(image_folder, file_name)
            try:
                images.append(imageio.imread(file_path))
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")

    print(f"Loaded {len(images)} images.")  # Debug: Check number of images loaded

    if not images:
        raise ValueError("No images found in the folder.")

    print(f"Saving GIF to: {gif_name}")  # Debug: Show where GIF is being saved
    imageio.mimsave(gif_name, images, fps=5)


images_to_gif(image_folder, gif_name)
