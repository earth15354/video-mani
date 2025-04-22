import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load and preprocess the image
image = mp.Image.create_from_file('../../Downloads/hard_demo.png')

# Perform hand landmark detection
detection_result = detector.detect(image)

# Draw landmarks on the image
annotated_image = image.numpy_view().copy()
for hand_landmarks in detection_result.hand_landmarks:
    for landmark in hand_landmarks:
        x = int(landmark.x * annotated_image.shape[1])
        y = int(landmark.y * annotated_image.shape[0])
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

# Display the annotated image
cv2.imshow('Hand Landmarks', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()