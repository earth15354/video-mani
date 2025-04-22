from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.hamer.configs import CACHE_DIR_HAMER
from hamer.hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.hamer.utils import recursive_to
from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full

# CACHE_DIR_HAMER = "./hamer/_DATA"
# CHANGED IN ./hamer/configs/__init__.py
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from hamer.vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

class HamerHand:
    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        img_folder='images',
        out_folder='out_demo',
        side_view=False,
        full_frame=True,
        save_mesh=False,
        batch_size=1,
        rescale_factor=2.0,
        body_detector='vitdet',
        file_type=['*.jpg', '*.png']
    ):
        self.checkpoint = checkpoint
        self.img_folder = img_folder
        self.out_folder = out_folder
        self.side_view = side_view
        self.full_frame = full_frame
        self.save_mesh = save_mesh
        self.batch_size = batch_size
        self.rescale_factor = rescale_factor
        self.body_detector = body_detector
        self.file_type = file_type

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        self.model_cfg = None
        self.detector = None
        self.cpm = None
        self.renderer = None

    
    def setup(self):
        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(self.checkpoint)

        # Setup HaMeR model
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load detector
        from hamer.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if self.body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer.hamer as hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif self.body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # keypoint detector
        self.cpm = ViTPoseModel(self.device)

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
    
    def _to_numpy(self, x):
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return x
    
    def get_hand_bbox(self, img_path):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            # print("No hands found in image")
            return [],[]

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Collect all vertices and camera translations
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t,
                render_res=img_size[n],
                is_right=all_right,
                **misc_args
            )

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            # Prepare image for bounding box drawing
            img_with_boxes = (input_img_overlay[:, :, ::-1] * 255).astype(np.uint8).copy()  # Convert to BGR uint8

            fx = fy = self._to_numpy(scaled_focal_length)
            cx, cy = self._to_numpy(img_size[n][0] / 2), self._to_numpy(img_size[n][1] / 2)

            bboxes = []
            # right = []
            for verts, cam_t in zip(all_verts, all_cam_t):
                # Convert to CPU numpy
                verts_np = self._to_numpy(verts)
                cam_t_np = self._to_numpy(cam_t)

                # Apply translation
                verts_cam = verts_np + cam_t_np  # (N, 3)
                x = self._to_numpy(verts_cam[:, 0])
                y = self._to_numpy(verts_cam[:, 1])
                z = self._to_numpy(verts_cam[:, 2])

                # Project to 2D
                u = (fx * x / z + cx).astype(np.int32)
                v = (fy * y / z + cy).astype(np.int32)

                # Bounding box
                x_min, x_max = np.clip(u.min(), 0, img_with_boxes.shape[1] - 1), np.clip(u.max(), 0, img_with_boxes.shape[1] - 1)
                y_min, y_max = np.clip(v.min(), 0, img_with_boxes.shape[0] - 1), np.clip(v.max(), 0, img_with_boxes.shape[0] - 1)

                bbox = [x_min, y_min, x_max, y_max]
                # right.append(right_hand)
                bboxes.append(bbox)
            return bboxes, all_right
        else:
            return [],[]

    def hand_bbox_demo(self):
        # Make output directory if it does not exist
        os.makedirs(self.out_folder, exist_ok=True)

        # Get all demo images ends with .jpg or .png
        img_paths = [img for end in self.file_type for img in Path(self.img_folder).glob(end)]

        # Iterate over all images in folder
        for img_path in img_paths:
            img_cv2 = cv2.imread(str(img_path))

            # Detect humans in image
            det_out = self.detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            vitposes_out = self.cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                keyp = right_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Collect all vertices and camera translations
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

            # Render front view
            if self.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = self.renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t,
                    render_res=img_size[n],
                    is_right=all_right,
                    **misc_args
                )

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

                # Prepare image for bounding box drawing
                img_with_boxes = (input_img_overlay[:, :, ::-1] * 255).astype(np.uint8).copy()  # Convert to BGR uint8

                fx = fy = self._to_numpy(scaled_focal_length)
                cx, cy = self._to_numpy(img_size[n][0] / 2), self._to_numpy(img_size[n][1] / 2)

                for verts, cam_t, right_hand in zip(all_verts, all_cam_t, all_right):
                    # Convert to CPU numpy
                    verts_np = self._to_numpy(verts)
                    cam_t_np = self._to_numpy(cam_t)

                    # Apply translation
                    verts_cam = verts_np + cam_t_np  # (N, 3)
                    x = self._to_numpy(verts_cam[:, 0])
                    y = self._to_numpy(verts_cam[:, 1])
                    z = self._to_numpy(verts_cam[:, 2])

                    # Project to 2D
                    u = (fx * x / z + cx).astype(np.int32)
                    v = (fy * y / z + cy).astype(np.int32)

                    # Bounding box
                    x_min, x_max = np.clip(u.min(), 0, img_with_boxes.shape[1] - 1), np.clip(u.max(), 0, img_with_boxes.shape[1] - 1)
                    y_min, y_max = np.clip(v.min(), 0, img_with_boxes.shape[0] - 1), np.clip(v.max(), 0, img_with_boxes.shape[0] - 1)

                    # Draw rectangle
                    box_color = (0, 0, 255) if right_hand == 0 else (255, 0, 0)
                    cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)

                # Save image with bounding boxes
                cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_all_boxed.png'), img_with_boxes)


    def full_demo(self):
        # Make output directory if it does not exist
        os.makedirs(self.out_folder, exist_ok=True)

        # Get all demo images ends with .jpg or .png
        img_paths = [img for end in self.file_type for img in Path(self.img_folder).glob(end)]

        # Iterate over all images in folder
        for img_path in img_paths:
            img_cv2 = cv2.imread(str(img_path))

            # Detect humans in image
            det_out = self.detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            vitposes_out = self.cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)
                
                multiplier = (2*batch['right']-1)
                pred_cam = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                multiplier = (2*batch['right']-1)
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    regression_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )

                    if self.side_view:
                        side_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                    cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                    # Save all meshes to disk
                    if self.save_mesh:
                        camera_translation = cam_t.copy()
                        tmesh = self.renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                        tmesh.export(os.path.join(self.out_folder, f'{img_fn}_{person_id}.obj'))
            
            # Render front view
            if self.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = self.renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t,
                    render_res=img_size[n],
                    is_right=all_right,
                    **misc_args
                )

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

                # Prepare image for bounding box drawing
                img_with_boxes = (input_img_overlay[:, :, ::-1] * 255).astype(np.uint8).copy()  # Convert to BGR uint8

                fx = fy = self._to_numpy(scaled_focal_length)
                cx, cy = self._to_numpy(img_size[n][0] / 2), self._to_numpy(img_size[n][1] / 2)

                for verts, cam_t in zip(all_verts, all_cam_t):
                    # Convert to CPU numpy
                    verts_np = self._to_numpy(verts)
                    cam_t_np = self._to_numpy(cam_t)

                    # Apply translation
                    verts_cam = verts_np + cam_t_np  # (N, 3)
                    x = self._to_numpy(verts_cam[:, 0])
                    y = self._to_numpy(verts_cam[:, 1])
                    z = self._to_numpy(verts_cam[:, 2])

                    
                    # Project to 2D
                    u = (fx * x / z + cx).astype(np.int32)
                    v = (fy * y / z + cy).astype(np.int32)

                    # Bounding box
                    x_min, x_max = np.clip(u.min(), 0, img_with_boxes.shape[1] - 1), np.clip(u.max(), 0, img_with_boxes.shape[1] - 1)
                    y_min, y_max = np.clip(v.min(), 0, img_with_boxes.shape[0] - 1), np.clip(v.max(), 0, img_with_boxes.shape[0] - 1)

                    # Draw rectangle
                    cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

                # Save image with bounding boxes
                cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_all_boxed.jpg'), img_with_boxes)

                # Save plain overlay image as before
                cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_all.jpg'), (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8))

if __name__ == '__main__':
    hamer = HamerHand(img_folder='./hamer/example_data', out_folder='hamer_test_out', batch_size=48)
    hamer.setup()
    # hamer.full_demo()
    hamer.hand_bbox_demo()
