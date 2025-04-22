import pyrealsense2 as rs
import cv2
import numpy as np
import os
from PIL import Image

class RealsenseBagExtractor:
    def __init__(self, bag_file, output_dir="extracted_frames", width=640, height=480, fps=30):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.color_dir = os.path.join(output_dir, "color")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.abs_depth_dir = os.path.join(output_dir, "depth_numpy")
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.abs_depth_dir, exist_ok=True)
        # self.width = width
        # self.height = height
        # self.fps = fps
        # os.makedirs(self.output_dir, exist_ok=True)

        # self.pipeline = rs.pipeline()
        # self.config = rs.config()
        # self.config.enable_device_from_file(self.bag_file, repeat_playback=False)
        # self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

    def extract_frames(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.bag_file, repeat_playback=False)

        # # Let the pipeline auto-configure from the bag file
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)

        profile = pipeline.start(config)

        for s in profile.get_streams():
            vsp = s.as_video_stream_profile()
            intr = vsp.get_intrinsics()
            print(f"Stream: {s.stream_type().name}")
            print(f"  Format: {vsp.format().name}")
            print(f"  Resolution: {intr.width}x{intr.height}")
            print(f"  FPS: {vsp.fps()}")
            print(f"  Intrinsics:")
            print(f"    Fx: {intr.fx}")
            print(f"    Fy: {intr.fy}")
            print(f"    Ppx (cx): {intr.ppx}")
            print(f"    Ppy (cy): {intr.ppy}")
            print(f"    Distortion Model: {intr.model}")
            print(f"    Coeffs: {intr.coeffs}")
            print("")

        # pipeline.stop()

        # align_to = rs.stream.color
        # align = rs.align(align_to)
        align = rs.align(rs.stream.color)
        frame_count = 0

        # frame_count = 0
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                if (frame_count%50) == 0:
                    print(f"[{frame_count}] Saved color and depth frames")
                    print(np.shape(color_image), np.shape(depth_image))

                color_image = color_image[:, :, [2, 1, 0]] # colors switched

                # save depth array
                np.save(os.path.join(self.abs_depth_dir, f"depth_{frame_count:05d}.npy"), depth_image)
                # # save color image
                # Image.fromarray(color_image).save(self.color_dir / f"color_{frame_count:05d}.png")

                # Save color image
                color_path = os.path.join(self.color_dir, f"color_{frame_count:05d}.png")
                cv2.imwrite(color_path, color_image)

                # Save depth image (16-bit PNG)
                depth_path = os.path.join(self.depth_dir, f"depth_{frame_count:05d}.png")
                cv2.imwrite(depth_path, depth_image)

                # print(f"[{frame_count}] Saved color and depth frames")
                frame_count += 1

        except RuntimeError:
            print("End of .bag file reached.")
        finally:
            pipeline.stop()
            # if all_depth:
            #     # Stack all depth frames into a 3D array and save
            #     stacked_depth = np.stack(all_depth, axis=0)
            #     print(np.shape(stacked_depth))
            #     np.save(self.abs_depth_dir, stacked_depth)
            #     print(f"Saved {frame_count} frames and depth array to '{self.output_dir}'")
            # else:
            #     print("No depth frames extracted.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a Realsense .bag file")
    parser.add_argument("--bag_file", type=str, required=True, help="Path to the .bag file")
    parser.add_argument("--output_dir", type=str, default="output_frames", help="Directory to save extracted frames")

    args = parser.parse_args()

    extractor = RealsenseBagExtractor(args.bag_file, args.output_dir)
    extractor.extract_frames()
