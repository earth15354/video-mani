import cv2
import os
import numpy as np

# Directory containing images
for j in range(4):
    image_dir = "./hand_object_detector/testdata"+str(j+1)
    output_dir = "./flow-outputs2/testoutflow"+str(j+1)  # Folder to save optical flow images
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(image_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)

    mags = []

    # Loop through image pairs and compute DeepFlow optical flow
    for i in range(1, len(image_files)):
        next_frame = cv2.imread(os.path.join(image_dir, image_files[i]), cv2.IMREAD_GRAYSCALE)

        # Compute optical flow using DeepFlow
        # flow = cv2.optflow.calcOpticalFlowDeepFlow(prev_frame, next_frame)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # print(np.shape(flow))
        # Convert flow to an RGB visualization
        # hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue: Direction
        # hsv[..., 1] = 255  # Saturation: Full
        # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value: Magnitude

        mags.append(magnitude)

    print(f"Done reading images for {image_dir}")

    max_value = np.max(np.array(mags))
    # print(max_value)

    for m in mags:
        # hsv = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv = m*255 / max_value
        # print(hsv[100][100])
        # Convert HSV to BGR for saving
        # flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_bgr = hsv

        # Save the optical flow image
        output_path = os.path.join(output_dir, f"flow_{i:04d}.png")
        cv2.imwrite(output_path, flow_bgr)

        # Update previous frame
        prev_frame = next_frame

    print(f"Optical flow images saved to {output_dir}")