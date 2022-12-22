import cv2
import dlib
import os

# Load the face detector and the shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capturer
cap = cv2.VideoCapture(0)

while True:
      # Read the frame from the webcam
  _, frame = cap.read()
  # Convert the frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Detect faces in the frame
  faces = face_detector(gray)
  # Iterate through the detected faces
  for face in faces:
    # Get the face landmarks
    landmarks = shape_predictor(gray, face)
    # Use the face landmarks to extract the face embedding
    face_embedding = face_recognition_model.compute_face_descriptor(frame, landmarks)
    # Predict the label
    label = classifier.predict([face_embedding])
    # Draw a bounding box around the face
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Put the label text above the bounding box
    cv2.putText(frame, label[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  # Display the frame
  cv2.imshow("Frame", frame)
  # Check if the user pressed 'q' to quit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
# Release the video capturer
cap.release()