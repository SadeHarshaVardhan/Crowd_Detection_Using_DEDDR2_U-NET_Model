#import the all requried packages..
#Step-1 start
import os
import numpy as np
import cv2
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
#Step-1 end
#step-2 start 
image_dir = "/content/drive/MyDrive/OLD512DATA-SET/OLD512IMAGES"
mat_dir = "/content/drive/MyDrive/OLD512DATA-SET/OLD512GT"

image_list = sorted(os.listdir(image_dir))
mat_list = sorted(os.listdir(mat_dir))

images = []
gt_masks = []

for img_name, mat_name in zip(image_list, mat_list):
    # Load image
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256))  # Resize as needed
    images.append(image)

    # Load corresponding .mat file
    mat_path = os.path.join(mat_dir, mat_name)
    mat = scipy.io.loadmat(mat_path)
    head_positions = mat['head_positions']  # shape: (N, 2)

    # Create blank GT mask
    gt_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for x, y in head_positions:
        x = int(x * 256 / image.shape[1])  # scale if needed
        y = int(y * 256 / image.shape[0])
        if 0 <= y < 256 and 0 <= x < 256:
            gt_mask[y, x] = 255  # mark the head point

    gt_masks.append(gt_mask)

images = np.array(images, dtype=np.float32) / 255.0
gt_masks = np.array(gt_masks, dtype=np.float32)
gt_masks = np.expand_dims(gt_masks, axis=-1)  # for compatibility with model
#step-2 end
#step-3 start
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.models import Model

# Convolutional block (2 Conv2D layers with ReLU activation)
def conv_block(x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    x1 = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    x1 = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x1)
    return x1

# Residual block: Adds a shortcut connection to help the learning process
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = tf.keras.activations.relu(x)
    return x

# Encoder block: Consists of a residual block followed by max pooling
def encoder_block(x, filters):
    x = residual_block(x, filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

# Decoder block: Upsampling followed by concatenation and residual block
def decoder_block(x, skip, filters):
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x

# Remove the recurrent block (ConvLSTM2D) for simplicity and use Conv2D instead.
def bridge_block(x, filters):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

# Build the DEDDRR U-Net model
def build_dedddrr_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder 1
    x1, p1 = encoder_block(inputs, 64)
    x2, p2 = encoder_block(p1, 128)
    x3, p3 = encoder_block(p2, 256)

    # Encoder 2
    x4, p4 = encoder_block(p3, 512)
    x5, p5 = encoder_block(p4, 512)

    # Bridge (Just a Conv2D layer here instead of ConvLSTM)
    bridge = bridge_block(p5, 1024)

    # Decoder 1
    d1 = decoder_block(bridge, x5, 512)
    d2 = decoder_block(d1, x4, 512)

    # Decoder 2
    d3 = decoder_block(d2, x3, 256)
    d4 = decoder_block(d3, x2, 128)
    d5 = decoder_block(d4, x1, 64)

    # Final output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d5)

    # Define the model
    model = Model(inputs, outputs, name="DEDDRR_UNet")
    return model

# Build the model with input shape of (256, 256, 3)
model = build_dedddrr_unet(input_shape=(256, 256, 3))

# Print the model summary to verify the architecture
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#step-3 end
#step-4 start
# Train the model with images and ground truth masks
model.fit(images, gt_masks, batch_size=16, epochs=50, validation_split=0.2)
#step-4 end
#step-5 start
# Save the model
model.save('/content/drive/MyDrive/DEDDR2/train1.keras')
#step-5 end
#step-6 start
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model_path = '/content/drive/MyDrive/DEDDR2/train1.keras'
model = load_model(model_path)

# Paths to test images and masks
test_images_path = '/content/drive/MyDrive/Pi(64)/PiImages'
test_masks_path = '/content/drive/MyDrive/Pi(64)/PiMask'

# Function to load and preprocess images
def load_images_and_masks(image_dir, mask_dir, image_size=(256, 256)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize
        images.append(img)

        # Load and preprocess mask
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
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on test images
predicted_masks = model.predict(test_images)

# Plot some results
def display_predictions(images, true_masks, pred_masks, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred_masks[i].squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Show predictions
display_predictions(test_images, test_masks, predicted_masks, num_samples=5)
#step-6 end
