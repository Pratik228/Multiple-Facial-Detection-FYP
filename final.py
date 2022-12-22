import cv2
import os
import numpy as np
import imutils

# Load the Caffe model
model_file = "model/deploy.prototxt.txt"
weights_file = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_file, weights_file)

# Read the images from the folders and label them with the person's name
folder_names = os.listdir("photos/")
images = {}
image_names = []
for folder_name in folder_names:
    images[folder_name] = []
    for file_name in os.listdir("photos/" + folder_name):
        file_path = "photos/" + folder_name + "/" + file_name
        image = cv2.imread(file_path)
        images[folder_name].append(image)
        image_names.append(folder_name)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()
    frame = imutils.resize(frame, width=400)

    # Pre-process the frame for the Caffe model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # Get the face detections
    detections = net.forward()

    # Loop through the detections and draw bounding boxes around the faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Recognize the face
            face = frame[startY:endY, startX:endX]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            net.setInput(face_blob)
            detections = net.forward()

            # Get the name of the person with the highest probability
            detections = detections[0, 0, :, 1]
            detection_index = np.argmax(detections)
            person_name = image_names[detection_index]

            # Put the label with the person's name above the bounding box
            label = "{}: {:.2f}%".format(person_name, confidence * 100)
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
