
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed facial images
data = np.load("processed_data.npz")

X = data["X"]
y = data["y"]

print("Loading facial data...")
print("Total images:", len(X))

# Convert identity labels into numerical labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# Create the SVM model
model = SVC(kernel="rbf", C=10, gamma="scale")

# Train the model
print("\nTraining facial recognition model...")
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Display classification results
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# Save the trained model and label encoder
joblib.dump(model, "face_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Model and label encoder saved successfully!")
predictions = model.predict(X_test)
# Save test images for recognition testing
np.savez_compressed(
    "test_data.npz",
    X_test=X_test,
    y_test=y_test
)