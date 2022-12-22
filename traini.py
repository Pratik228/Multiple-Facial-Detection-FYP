# import cv2
# import os

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# # Set the root directory for the photos
# root_dir = 'photos'

# # Create a dictionary to map subfolder names to person names
# person_names = {}

# # Iterate over the subfolders in the root directory
# for subdir in os.listdir(root_dir):
#     subdir_path = os.path.join(root_dir, subdir)
#     if os.path.isdir(subdir_path):
#         # Get the name of the person from the subfolder name
#         person_name = subdir
#         person_names[subdir] = person_name

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # Iterate over the faces and draw a rectangle and text on the frame
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Get the name of the person from the subfolder name
#         if subdir in person_names:
#             person_name = person_names[subdir]
#             cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Show the frame
#     cv2.imshow('frame', frame)

#     # Check for user input
#     key = cv2.waitKey(1)
#     if key == 27:  # Press 'ESC' to exit
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import dlib
# import os

# # Load the pre-trained face detection model
# face_detector = dlib.get_frontal_face_detector()

# # Set the root directory for the photos
# root_dir = 'photos'

# # Create a dictionary to map subfolder names to person names
# person_names = {}

# # Iterate over the subfolders in the root directory
# for subdir in os.listdir(root_dir):
#     subdir_path = os.path.join(root_dir, subdir)
#     if os.path.isdir(subdir_path):
#         # Get the name of the person from the subfolder name
#         person_name = subdir
#         person_names[subdir] = person_name

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_detector(gray, 0)

#     # Iterate over the faces and draw a rectangle and text on the frame
#     for face in faces:
#         x, y, w, h = face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top()
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Get the name of the person from the subfolder name
#         if subdir in person_names:
#             person_name = person_names[subdir]
#             cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Show the frame
#     cv2.imshow('frame', frame)

#     # Check for user input
#     key = cv2.waitKey(1)
#     if key == 27:  # Press 'ESC' to exit
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mtcnn
# import os

# # Load the pre-trained face detection model
# face_detector = mtcnn.MTCNN()

# # Set the root directory for the photos
# root_dir = 'photos'

# # Create a dictionary to map subfolder names to person names
# person_names = {}

# # Iterate over the subfolders in the root directory
# for subdir in os.listdir(root_dir):
#     subdir_path = os.path.join(root_dir, subdir)
#     if os.path.isdir(subdir_path):
#         # Get the name of the person from the subfolder name
#         person_name = subdir
#         person_names[subdir] = person_name

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces in the frame
#     faces = face_detector.detect_faces(frame_rgb)

#     # Iterate over the faces and draw a rectangle and text on the frame
#     for face in faces:
#         x, y, w, h = face['box']
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Get the name of the person from the subfolder name
#         if subdir in person_names:
#             person_name = person_names[subdir]
#             cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Show the frame
#     cv2.imshow('frame', frame)

#     # Check for user input
#     key = cv2.waitKey(1)
#     if key == 27:  # Press 'ESC' to exit
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import imutils

# Load the Caffe model
model_file = "model/deploy.prototxt.txt"
weights_file = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_file, weights_file)

# Set the input image dimensions
# img_width, img_height = (224, 224)

# Read the images from the folders and label them with the person's name
folder_names = os.listdir("photos/")
images = {}
for folder_name in folder_names:
    images[folder_name] = []
    for file_name in os.listdir("photos/" + folder_name):
        file_path = "photos/" + folder_name + "/" + file_name
        image = cv2.imread(file_path)
        images[folder_name].append(image)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()
    frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
    (img_height, img_width) = frame.shape[:2]

    # Pre-process the frame for the Caffe model
    #blob = cv2.dnn.blobFromImage(frame, 1.0, (img_width, img_height), (104, 117, 123))
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    

    # Get the face detections
    detections = net.forward()

    # Loop through the detections and draw bounding boxes around the faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Put the label with the person's name above the bounding box
            label = "{}: {:.2f}%".format(folder_name, confidence * 100)
            # y = startY - 10 if startY > 20 else startY + 10
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    # Show the output frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    # Break the loop if the user presses the "q" key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture object and destroy the windows
cap.release()
cv2.destroyAllWindows()