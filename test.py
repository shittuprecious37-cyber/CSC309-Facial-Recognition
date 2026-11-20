
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and label encoder
model = joblib.load("face_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load the test images
data = np.load("test_data.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# Display available test images
print("Available test images:", len(X_test))

# Ask the user to select an image
choice = int(input("Enter an image number (1-40): "))

if choice < 1 or choice > len(X_test):
    print("Invalid selection. Choose a number between 1 and 40.")
else:
    # Get the selected image
    index = choice - 1
    image = X_test[index]

    # Predict the person's identity
    prediction = model.predict(image.reshape(1, -1))
    predicted_person = encoder.inverse_transform(prediction)[0]

    # Get the actual identity
    actual_person = encoder.inverse_transform([y_test[index]])[0]

    print("\nFACIAL RECOGNITION RESULT")
    print("------------------------")
    print("Predicted identity:", predicted_person)
    print("Actual identity:", actual_person)

    if predicted_person == actual_person:
        print("Result: Correct recognition!")
    else:
        print("Result: Incorrect recognition.")

    # Display the selected face
    plt.imshow(image.reshape(64, 64), cmap="gray")
    plt.title("Predicted: " + predicted_person)
    plt.axis("off")
    plt.show()

