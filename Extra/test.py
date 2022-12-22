import face_recognition
import os
import cv2

# Load the list of known face names and encodings
known_face_names = []
known_face_encodings = []

# Loop through the photos folder and its subfolders
for root, dirs, files in os.walk('photos/tag'):
    # Loop through the files in the current subfolder
    for file in files:
        # Load the image
        image = face_recognition.load_image_file(os.path.join(root, file))

        # Get the face encoding for the image
        face_encoding = face_recognition.face_encodings(image)[0]

        # Get the name of the subfolder (which is the name of the person)
        name = os.path.split(root)[1]

        # Add the name and encoding to the lists
        known_face_names.append(name)
        known_face_encodings.append(face_encoding)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Detect the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through the detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face is a match for a known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match was found, tag the name of the person
        if True in matches:
            name = known_face_names[matches.index(True)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame with the detected and tagged faces
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all
cv2.destroyAllWindows()