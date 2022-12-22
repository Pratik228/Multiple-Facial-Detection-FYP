import cv2
import os
import mtcnn
import numpy as np
# Load the pre-trained face detection model
model = mtcnn.MTCNN()

# Create a dictionary to map subfolder names to person names
person_names = {}

# Set the root directory for the photos
root_dir = 'photos'

# Iterate over the subfolders in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        # Get the name of the person from the subfolder name
        person_name = subdir
        person_names[subdir] = person_name

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)
    mean = np.mean(frame)
    std = np.std(frame)
    frame = (frame - mean) / std
    frame = np.expand_dims(frame, axis=0)

    # Make a prediction using the trained model
    predictions = model.predict