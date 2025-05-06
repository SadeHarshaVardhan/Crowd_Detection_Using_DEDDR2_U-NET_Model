import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

# ----- 1. Load Trained Model -----
model_path = r'C:\Users\Gautam Pothurajula\Downloads\DEDCWNET22.keras'

# Custom loss function (if required)
def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the model
loaded_model = load_model(model_path, safe_mode=False, custom_objects={'custom_loss': custom_loss})

# ----- 2. Define Live Video Capture -----
video_url = "http://192.168.1.12:5000/video"
save_dir = r"D:\LiveStream"
os.makedirs(save_dir, exist_ok=True)

image_path = os.path.join(save_dir, "latest_frame.jpg")  # Constant file name to avoid storage issues

print("Connecting to video stream...")
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video stream!")
    exit()

print("Successfully connected to video stream.")

# ----- 3. Function to Load and Preprocess Image -----
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Loads and preprocesses the latest captured image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image in color mode
    if img is None:
        print("Warning: Unable to load image.")
        return None
    
    img_resized = cv2.resize(img, target_size)  # Resize to model input size
    img_normalized = img_resized / 255.0  # Normalize to [0,1]

    return np.expand_dims(img_normalized, axis=0).astype(np.float32)  # Add batch dimension

# ----- 4. Function to Perform Model Prediction -----
def predict_image(image):
    """Performs model prediction and returns thresholded results."""
    if image is None:
        return None
    preds = loaded_model.predict(image)
    return (preds > 0.3).astype(int)  # Apply thresholding

# ----- 5. Function to Display Prediction -----
def display_prediction(input_img, prediction):
    """Displays the input image alongside its prediction with timestamp."""
    if prediction is None:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.figure(figsize=(10, 5))

    # Input Image
    plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.title(f"Input Image\n{timestamp}")
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.imshow(prediction[0], cmap='gray')  # Show first image (batch size = 1)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

# ----- 6. Continuous Live Processing Loop -----
while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to read frame from video stream.")
        break

    # Overwrite the existing image
    cv2.imwrite(image_path, frame)
    print(f"Saved: {image_path}")

    # Load and process image
    processed_image = load_and_preprocess_image(image_path)

    # Perform prediction
    preds = predict_image(processed_image)

    # Display result
    display_prediction(processed_image[0], preds)

    time.sleep(1)  # Capture one frame per second

cap.release()
print("Live processing ended.")
