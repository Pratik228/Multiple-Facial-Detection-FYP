import os
import numpy as np
import cv2
import dlib
import tensorflow as tf

# Load the DCNN model with the ImageNet weights
model = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

def preprocess_image(image_path):
      # Load the image
  image = cv2.imread(image_path)
  # Resize the image to the desired size
  image = cv2.resize(image, (224, 224))
  # Convert the image to a float array
  image = image.astype(np.float32)
  # Normalize the image
  image /= 255.0
  # Return the preprocessed image
  return image

def generate_data(photos):
      # Initialize the lists for the images and labels
  images = []
  labels = []
  # Iterate through the subfolders
  for subfolder in os.listdir(photos):
    subfolder_path = os.path.join(photos, subfolder)
    # Iterate through the images in the subfolder
    for image_path in os.listdir(subfolder_path):
      # Load the image and the label
      image = preprocess_image(os.path.join(subfolder_path, image_path))
      # Remove the extra dimension
      image = image[0]
      label = subfolder
      # Append the label and image to the lists
      labels.append(label)
      images.append(image)
  # Return the lists as a tuple
  return np.array(images), np.array(labels)

# Load the data and split it into train and test sets
from sklearn.model_selection import train_test_split
X, y = generate_data("photos")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a new model using the tf.keras library
new_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(os.listdir("photos")), activation='softmax')
])

# Compile the model
new_model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
loss=tf.keras.losses.CategoricalCrossentropy(),
metrics=['accuracy'])
new_model.fit(X_train, y_train, epochs=10)
loss, accuracy = new_model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)