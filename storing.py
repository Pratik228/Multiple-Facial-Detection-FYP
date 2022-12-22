import cv2
import os

# Get the name of the person from the user
name = input("Enter the name of the person: ")
# Create a folder to store the photos of the person
folder_name = "photos/" + name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the counter for the photos
i = 0

# Take multiple photos
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the user clicked the left mouse button
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Space bar is ASCII 32
        # Save the photo in the folder
        file_name = folder_name + "/" + name + str(i) + ".jpg"
        cv2.imwrite(file_name, frame)
        print("Saved photo as", file_name)
        i += 1  # Increment the counter

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()