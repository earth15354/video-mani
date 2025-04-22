import os
import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge

def extract_images(bag_path, rgb_topic, depth_topic, rgb_out_dir, depth_out_dir):
    bag = rosbag.Bag(bag_path, "r")
    bridge = CvBridge()

    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)

    rgb_count = 0
    depth_count = 0

    for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic]):
        if topic == rgb_topic:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = os.path.join(rgb_out_dir, f"frame_{rgb_count:04d}.png")
            cv2.imwrite(filename, cv_image)
            rgb_count += 1
        elif topic == depth_topic:
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_filename = os.path.join(depth_out_dir, f"depth_{depth_count:04d}.npy")
            np.save(depth_filename, depth_image.astype(np.float32))
            depth_count += 1

    bag.close()

    print(f"Saved {rgb_count} RGB frames to {rgb_out_dir}")
    print(f"Saved {depth_count} depth frames to {depth_out_dir}")

if __name__ == "__main__":
    extract_images(
        bag_path="./rosbags/test_demo.bag",
        rgb_topic="/multisense/left/image_color",
        depth_topic="/multisense/depth",
        rgb_out_dir="./rosbags/rgb_frames",
        depth_out_dir="./rosbags/depth_frames"
    )
