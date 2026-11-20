import os
import cv2
from sklearn.datasets import fetch_olivetti_faces

# Download the facial image dataset
print("Downloading facial image dataset...")

faces = fetch_olivetti_faces()

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Select 20 people, with 10 images each
for i in range(200):

    image = faces.images[i]
    label = faces.target[i]

    # Create a folder for each person
    person_folder = f"dataset/person_{label}"
    os.makedirs(person_folder, exist_ok=True)

    # Convert image into a format OpenCV can save
    image = (image * 255).astype("uint8")

    # Save facial image
    image_path = f"{person_folder}/face_{i}.jpg"
    cv2.imwrite(image_path, image)

print("Successfully downloaded and saved 200 facial images!")