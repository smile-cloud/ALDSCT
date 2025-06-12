import os
import random
import shutil
from tqdm import tqdm

# Specify the path to the images and labels folders
image_dir = "./data/images"
label_dir = "./data/labels"

# Create lists to store valid images and labels paths
valid_images = []
valid_labels = []

# Iterate through all images in the image directory
for image_name in os.listdir(image_dir):
    # Get the full path of the image
    image_path = os.path.join(image_dir, image_name)

    # Extract the file extension and replace it to match label filenames
    ext = os.path.splitext(image_name)[-1]
    label_name = image_name.replace(ext, ".txt")
    label_path = os.path.join(label_dir, label_name)

    # Check if the corresponding label exists
    if os.path.exists(label_path):
        # Add paths to the lists for later processing
        valid_images.append(image_path)
        valid_labels.append(label_path)
    else:
        # Optional: Print if any images are missing corresponding labels
        print("Missing label for:", image_path)

# Destination directories
train_dir = "../datasets/20241106/train"
valid_dir = "../datasets/20241106/valid"
test_dir = "../datasets/20241106/test"

# Create folders for train/valid/test images and labels if they don't exist
for folder in [train_dir, valid_dir, test_dir]:
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

# Process each valid image and label pair
for i in tqdm(range(len(valid_images))):
    image_path = valid_images[i]
    label_path = valid_labels[i]

    # Randomly assign the files to train, validation, or test sets
    r = random.random()
    if r < 0.1:
        destination = test_dir
    elif r < 0.3:
        destination = valid_dir
    else:
        destination = train_dir

    # Define the destination paths for images and labels
    image_destination_path = os.path.join(destination, "images", os.path.basename(image_path))
    label_destination_path = os.path.join(destination, "labels", os.path.basename(label_path))

    # Copy images and labels to the designated destination
    shutil.copy2(image_path, image_destination_path)
    shutil.copy2(label_path, label_destination_path)

print("Total valid images:", len(valid_images))
print("Total valid labels:", len(valid_labels))
