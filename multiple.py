import cv2
import face_recognition

# Initialize the webcam and the face count
cap = cv2.VideoCapture(0)
face_count = 0

# Loop indefinitely
while True:
    # Capture a frame from the webcam
    _, frame = cap.read()
    
    # Find all the faces in the frame using the face_recognition library
    face_locations = face_recognition.face_locations(frame)
    
    # Increment the face count
    
    # Draw a rectangle around each face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Webcam', frame)
    
    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()