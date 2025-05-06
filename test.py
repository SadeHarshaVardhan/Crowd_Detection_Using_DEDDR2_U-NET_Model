import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Path to trained model
model_path = '/content/drive/MyDrive/DEDDR2/train1.keras'
model = load_model(model_path)

# Paths to test images and masks
test_images_path = '/content/drive/MyDrive/Pi(64)/PiImages'
test_masks_path = '/content/drive/MyDrive/Pi(64)/PiMask'

# Function to load and preprocess images and masks
def load_images_and_masks(image_dir, mask_dir, image_size=(256, 256)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)

        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size)
        mask = mask / 255.0  # Normalize
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load test data
test_images, test_masks = load_images_and_masks(test_images_path, test_masks_path)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_masks)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on test images
predicted_masks = model.predict(test_images)

# Function to display predictions with crowd counts
def display_predictions_with_counts(images, true_masks, pred_masks, num_samples=5):
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(14, 4))

        # Compute counts
        pred_count = np.sum(pred_masks[i])
        gt_count = np.sum(true_masks[i])

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')

        # Ground truth mask with count
        plt.subplot(1, 3, 2)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title(f"Ground Truth Mask\nCount: {gt_count:.2f}")
        plt.axis('off')

        # Predicted mask with count
        plt.subplot(1, 3, 3)
        plt.imshow(pred_masks[i].squeeze(), cmap='gray')
        plt.title(f"Predicted Mask\nCount: {pred_count:.2f}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Show predictions
display_predictions_with_counts(test_images, test_masks, predicted_masks, num_samples=5)
