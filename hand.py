import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os
import math

class Hand:
    def __init__(self, input_pth):
        self.input_pth = input_pth
        self.annotated_image = None
    
    def get_hand_keypoints(self):
        # Initialize the Hand Landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)

        # Load and preprocess the image
        image = mp.Image.create_from_file(self.input_pth)

        # Perform hand landmark detection
        result = detector.detect(image)

        # Draw landmarks on the image
        annotated_image = image.numpy_view().copy()
        max_xl = 0
        max_yl = 0
        min_xl = annotated_image.shape[1]
        min_yl = annotated_image.shape[0]
        max_xr = 0
        max_yr = 0
        min_xr = annotated_image.shape[1]
        min_yr = annotated_image.shape[0]
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                if label == "Left":
                    if x > max_xl:
                        max_xl = x
                    if x < min_xl:
                        min_xl = x
                    if y > max_yl:
                        max_yl = y
                    if y < min_yl:
                        min_yl = y
                if label == "Right":
                    if x > max_xr:
                        max_xr = x
                    if x < min_xr:
                        min_xr = x
                    if y > max_yr:
                        max_yr = y
                    if y < min_yr:
                        min_yr = y 
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        cv2.rectangle(annotated_image, (min_xr, min_yr), (max_xr, max_yr), color=(255,0,0), thickness=2)
        cv2.rectangle(annotated_image, (min_xl, min_yl), (max_xl, max_yl), color=(0,0,255), thickness=2)
        
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        # Display the annotated image
        self.annotated_image = annotated_image
        
        left = (min_xl, min_yl, max_xl, max_yl)
        right = (min_xr, min_yr, max_xr, max_yr)
        return left, right

    def show_image(self):
        cv2.imshow('Hand Landmarks', self.annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hand_script.py <hand_name>")
        sys.exit(1)

    path = sys.argv[1]
    obj = Hand(path)
    obj.get_hand_keypoints()
    obj.show_image()