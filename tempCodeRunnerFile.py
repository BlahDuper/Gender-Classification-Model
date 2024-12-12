from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load the model
model = load_model('gender_detection.keras')

# Classes for gender detection
classes = ['man', 'woman']

# Path to the folder containing images
folder_path = '.\\archive\\Dataset\\Validation'

# Assuming there's only one frame (image) to process
subfolder = 'Male'  # You can change this based on the specific requirement
image_file = os.listdir(os.path.join(folder_path, subfolder))[0]

# Construct the full image path
image_path = os.path.join(folder_path, subfolder, image_file)

# Read the image
frame = cv2.imread(image_path)

if frame is None:
    print(f"Failed to read image: {image_file}")
else:
    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Use only the first detected face if multiple faces are detected
    if len(faces) > 0:
        f = faces[0]

        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over the detected face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small face regions
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            print("Detected face is too small.")
            cv2.imshow("Gender Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()

        # Preprocess the face for the gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on the face
        conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, e.g., [[0.9999, 0.0001]]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        print(label)
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Wait for a key press to close the window
    cv2.waitKey(0)

# Release resources
cv2.destroyAllWindows()
