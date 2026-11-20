
import cv2
import numpy as np
from pathlib import Path

# Locate the project and dataset folders
project_folder = Path(__file__).resolve().parent
dataset_folder = project_folder / "dataset"

# Store processed images and their labels
images = []
labels = []

print("Starting image preprocessing...")

# Go through each person's folder
for person_folder in sorted(dataset_folder.iterdir()):

    if not person_folder.is_dir():
        continue

    # Read every facial image
    for image_file in sorted(person_folder.glob("*.jpg")):

        image = cv2.imread(str(image_file))

        if image is None:
            print("Could not read:", image_file)
            continue

        # Resize image to 64 x 64 pixels
        image = cv2.resize(image, (64, 64))

        # Convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values from 0-255 to 0-1
        image = image.astype("float32") / 255.0

        # Flatten image into a one-dimensional array
        image = image.flatten()

        # Store image and its identity
        images.append(image)
        labels.append(person_folder.name)

# Convert lists into NumPy arrays
X = np.array(images)
y = np.array(labels)

# Save processed data
np.savez_compressed(
    project_folder / "processed_data.npz",
    X=X,
    y=y
)

print("\nPreprocessing completed successfully!")
print("Total images processed:", len(X))
print("Image data shape:", X.shape)
print("Labels shape:", y.shape)
