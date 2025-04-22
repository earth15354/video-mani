from hand_hamer import HamerHand
from object import ObjectDINO
import os
import shutil
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class VideoMani1:
    def __init__(self, in_folder, out_folder, text_prompt, depth_folder=None, file_type=['*.jpg', '*.png']):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.hand_det = HamerHand()
        self.object_det = ObjectDINO(text_prompt)
        self.file_type = file_type
        self.depth_folder = depth_folder
        # self.intrinsics = {
        #     'fx': 607.666,
        #     'fy': 607.388,
        #     'cx': 325.479,
        #     'cy': 245.907
        # }
        self.intrinsics = {
            'fx': 600.3658,
            'fy': 600.3335,
            'cx': 509.7308,
            'cy': 296.8633
        }
        R = np.array([  # Flattened 3x3 rotation matrix
            0.9999937415122986,  0.003299968084320426,  0.001276906463317573,
            -0.0032986891455948353, 0.9999940395355225, -0.0010023434879258275,
            -0.001280206604860723, 0.0009981251787394285, 0.9999986886978149
        ])
        T = np.array([0.0, 0.0, 1.0]) # made up
        self.extrinsics = {
            'R': R.reshape(3, 3),                     
            't': T    
        }

    def full_demo(self):
        self.hand_det.setup()
        self.object_det.load_model()

        # Make output directory if it does not exist
        os.makedirs(self.out_folder, exist_ok=True)
        
        # Get all demo images ends with .jpg or .png
        img_paths = sorted([img for end in self.file_type for img in Path(self.in_folder).glob(end)])
        depth_paths = None
        if self.depth_folder != None:
            depth_paths = sorted([depth for end in ['*.npy'] for depth in Path(self.depth_folder).glob(end)])

        frame = 1
        contact_informationl = []
        contact_informationr = []
        all_hand_bboxes = []
        all_all_right = []
        all_obj_bboxes = []
        all_labels = []
        # all_depth_values = []
        all_img_pil = []
        for i,img_path in enumerate(img_paths):
            img_pil, img = self.object_det.load_image(img_path)
            hand_bboxes, all_right = self.hand_det.get_hand_bbox(img_path)
            object_bboxes, labels = self.object_det.get_scaled_grounding_output(img, img_pil)
            
            contactl = 0
            contactr = 0
            for oi, obj in enumerate(object_bboxes):
                for hand, right in zip(hand_bboxes, all_right):
                    if (self.bbox_intersects(hand, obj, 0.5)):
                        print("CONTACT! ", frame)
                        if right:
                            contactr = 1
                        else:
                            contactl = 1
            contact_informationr.append(contactr)
            contact_informationl.append(contactl)
            all_hand_bboxes.append(hand_bboxes)
            all_all_right.append(all_right)
            all_obj_bboxes.append(object_bboxes)
            all_labels.append(labels)
            all_img_pil.append(img_pil)

            print("Read Frame: ", frame)
            frame += 1
        
        contact_informationl = self.rolling_median_smoothing(contact_informationl, 0.2)
        contact_informationr = self.rolling_median_smoothing(contact_informationr, 0.2)

        suml = sum(contact_informationl)
        sumr = sum(contact_informationr)

        if sumr >= suml:
            print("Using right hand")
            contact_information = contact_informationr
            right_local = 1
        else:
            print("Using left hand")
            contact_information = contact_informationl
            right_local = 0

        # contact_information = np.array(contact_information, dtype=bool)
        # print(np.shape(contact_information))
        # list(range(len(contact_information)))
        contact_hand_bboxes = []
        contact_all_right = []
        contact_obj_bboxes = []
        contact_labels = []
        contact_depth_values = []
        contact_img_pil = []
        for i,b in enumerate(contact_information):
            if b:
                # contact_information.append(contact)
                contact_hand_bboxes.append(all_hand_bboxes[i])
                contact_all_right.append(all_all_right[i])
                contact_obj_bboxes.append(all_obj_bboxes[i])
                contact_labels.append(all_labels[i])
                contact_img_pil.append(all_img_pil[i])
                if depth_paths != None:
                    depth_path = depth_paths[i]
                    depth_np = np.load(depth_path)
                    contact_depth_values.append(depth_np)
            # print(i, ". bool")

        # Make a list of ascending frame numbers and mask with contact_information binary list
        contact_information = np.array(contact_information, dtype=bool)
        contact_frames = np.array(list(range(len(contact_information))))[contact_information]
        # print(contact_frames)

        shutil.rmtree(self.out_folder)
        os.makedirs(self.out_folder)
    
        positions = []
        print("Storing Frames") 
        for i, img_pil in enumerate(contact_img_pil):
            frame = contact_frames[i]
            img_pil_copy = img_pil.copy()
            draw = ImageDraw.Draw(img_pil_copy)
            obj_color = (0,255,0)
            right_color = (255,0,0)
            left_color = (0,0,255)

            object_bboxes = contact_obj_bboxes[i]
            labels = contact_labels[i]
            hand_bboxes = contact_hand_bboxes[i]
            all_right = contact_all_right[i]

            for obj, label in zip(object_bboxes, labels):
                draw.rectangle(obj, outline=obj_color, width=2)
                # print("FISHY: ", obj)
            
            for i, (hand, right) in enumerate(zip(hand_bboxes, all_right)):
                if right:
                    draw.rectangle(hand, outline=right_color, width=2)
                else:
                    draw.rectangle(hand, outline=left_color, width=2)
                
                if right == right_local:
                    xp = int((hand[0]+hand[2])/2)
                    yp = int((hand[3]+hand[1])/2)
                    depth_np = None
                    if len(contact_depth_values) != 0:
                        depth_np = contact_depth_values[i]
                        # position = (xp, yp, depth_np[yp,xp])
                        # print(np.shape(depth_np))
                        depth_np = self.find_valid_depth(depth_np, xp, yp, 1000)
                        world_point = self.pixel_to_world(xp, yp, depth_np, None, None, camera_frame=True) # comment out for pixel diagram
                        world_point = world_point
                        positions.append(world_point)

            img_pil_copy.save(os.path.join(self.out_folder, "pred_"+str(frame)+".jpg"))
            # print("Stored Frame: ", frame)
        # print(positions)
        np.save(os.path.join(self.out_folder, "positions.npy"), positions)
        self.plot_trajectory(np.array(positions))
        return np.array(positions)
    
    def find_valid_depth(self, depth_np, x, y, max_radius=5):
        h, w = depth_np.shape
        x, y = int(round(x)), int(round(y))
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        value = depth_np[ny, nx]
                        if not np.isnan(value) and value != 0:
                            return value
        return np.nan

    def pixel_to_world(self, x, y, depth, intrinsics, extrinsics, camera_frame=True):
        """
        x, y: pixel coordinates
        depth: depth at (x, y)
        intrinsics: dict with fx, fy, cx, cy
        extrinsics: dict with R (3x3) and t (3,)
        """
        if intrinsics==None:
            intrinsics = self.intrinsics
        if extrinsics==None:
            if camera_frame:
                extrinsics = {
                    'R': np.eye(3),                     
                    't': np.zeros((3,))   
                }
            else:
                extrinsics = self.extrinsics

        # x,y,depth = position
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        R, t = extrinsics['R'], extrinsics['t']  # assume R, t are camera-to-world

        # Step 1: Pixel -> Camera Space
        Xc = (x - cx) * depth / fx
        Yc = (y - cy) * depth / fy
        Zc = depth
        cam_point = np.array([Xc, Yc, Zc])

        # Step 2: Camera -> World Space
        world_point = R @ cam_point + t
        # print(type(world_point))
        return np.array(world_point)

    
    def test_shapes(self):
        img_paths = sorted([img for end in self.file_type for img in Path(self.in_folder).glob(end)])
        depth_paths = None
        # img_paths = np.array(img_paths)
        if self.depth_folder != None:
            depth_paths = sorted([depth for end in ['*.npy'] for depth in Path(self.depth_folder).glob(end)])
            # depth_paths = np.array(depth_paths)
            # print("Depth shape: ", np.shape(depth_paths))
        
        img_nps = []
        depth_nps = []
        count = 0
        for i,img_path in enumerate(img_paths):
            img_pil, img = self.object_det.load_image(img_path)
            if depth_paths != None:
                depth_path = depth_paths[i]
                depth_np = np.load(depth_path)
                depth_nps.append(depth_np)
            # print(type(img_pil))
            # print(type(img))
            img_np = img.numpy()
            img_nps.append(img_np)
            if (count % 25) == 0:
                print(np.shape(img_np))
                print(count)
            count+=1

        print("Image shape: ", np.shape(img_paths))
        if self.depth_folder != None:
            print("Depth shape: ", np.shape(depth_nps))

    def plot_trajectory(self, points):
        if len(points) < 2:
            print("Not enough points to plot trajectory.")
            return

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Create segments for Line3DCollection
        segments = np.array([[points[i], points[i + 1]] for i in range(len(points) - 1)])

        # Normalize time steps for colormap
        t = np.linspace(0, 1, len(segments))
        colors = plt.cm.plasma(t)  # you can try 'viridis', 'cool', 'jet', etc.

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the colored line segments
        lc = Line3DCollection(segments, colors=colors, linewidths=2)
        ax.add_collection3d(lc)

        # Equal scaling
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Over Time (Blue → Red)')

        plt.show()

    
    def demo2d(self):
        self.hand_det.setup()
        self.object_det.load_model()

        # Make output directory if it does not exist
        os.makedirs(self.out_folder, exist_ok=True)
        
        # Get all demo images ends with .jpg or .png
        img_paths = sorted([img for end in self.file_type for img in Path(self.in_folder).glob(end)])

        frame = 1
        for img_path in img_paths:
            img_pil, img = self.object_det.load_image(img_path)
            hand_bboxes, all_right = self.hand_det.get_hand_bbox(img_path)
            object_bboxes, labels = self.object_det.get_scaled_grounding_output(img, img_pil)
            

            img_pil_copy = img_pil.copy()
            draw = ImageDraw.Draw(img_pil_copy)
            obj_color = (0,255,0)
            right_color = (255,0,0)
            left_color = (0,0,255)

            for obj, label in zip(object_bboxes, labels):
                draw.rectangle(obj, outline=obj_color, width=2)
                # print("FISHY: ", obj)
            
            for hand, right in zip(hand_bboxes, all_right):
                if right:
                    draw.rectangle(hand, outline=right_color, width=2)
                else:
                    draw.rectangle(hand, outline=left_color, width=2)

            img_pil_copy.save(os.path.join(self.out_folder, "pred_"+str(frame)+".jpg"))
            print("Finished Frame: ", frame)
            frame += 1

    def bbox_intersects(self, bbox1, bbox2, threshold=0.1):
        """
        Check if two bounding boxes intersect past a certain threshold.

        Parameters:
            bbox1 (list): Bounding box 1 in the format [x0, y0, x1, y1].
            bbox2 (list): Bounding box 2 in the format [x0, y0, x1, y1].
            threshold (float): Threshold (in range 0 to 1) for the intersection area relative to the area of the smaller box.

        Returns:
            bool: True if the intersection area exceeds the threshold, False otherwise.
        """

        # Unpack bounding boxes
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # Calculate the coordinates of the intersection box
        x0_int = max(x0_1, x0_2)
        y0_int = max(y0_1, y0_2)
        x1_int = min(x1_1, x1_2)
        y1_int = min(y1_1, y1_2)

        # Check if there is an intersection
        if x1_int <= x0_int or y1_int <= y0_int:
            return False  # No intersection

        # Calculate areas
        area_bbox1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area_bbox2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        area_intersection = (x1_int - x0_int) * (y1_int - y0_int)

        # Find the smaller area
        smaller_area = min(area_bbox1, area_bbox2)

        # Calculate intersection ratio
        intersection_ratio = area_intersection / smaller_area

        # Check if intersection exceeds threshold
        return intersection_ratio >= threshold

    def rolling_median_smoothing(self, data, scale=0.1):
        data = np.array(data)
        window_size = max(3, int(len(data) * scale))  # scale = fraction of total length
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        smoothed = pd.Series(data).rolling(window_size, center=True, min_periods=1).median()
        return smoothed.round().astype(int).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Path to folder with input images')
    parser.add_argument('--output', type=str, required=True, help="Path to folder with output images")
    parser.add_argument('--text', type=str, required=True, help='Text prompt for GroundingDINO (word to describe object)')
    parser.add_argument('--depth_path', type=str, default=None, help='Path to depth .npy file')      

    args = parser.parse_args()

    # depth_path = './extracted_bag/demo2/depth_numpy' # folder with numpy arrays of each depth image
    depth_path = args.depth_path
    vidmani = VideoMani1(args.input, args.output, args.text)
    vidmani_depth = VideoMani1(args.input, args.output, args.text, depth_folder=depth_path)
    # vidmani.demo2d() 
    # vidmani.full_demo()   
    # vidmani_depth.full_demo()
    vidmani_depth.test_shapes()

# ./epic-kitchens+fields/clipped/28.take_spatula/
# ../../Downloads/oven_demo
# ../../Downloads/box_demo
# ../../Downloads/cabinet_demo



