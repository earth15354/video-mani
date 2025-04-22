import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.vl_utils import create_positive_map_from_span

config_file_default = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
checkpoint_path_default = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
image_path_default = './GroundingDINO/images/fridge_ego.png'
output_dir_default = './grounding_outputs'

class ObjectDINO:
    def __init__(
            self,
            text_prompt: str,
            image_path: str = image_path_default,
            config_file: str = config_file_default,
            checkpoint_path: str = checkpoint_path_default,
            output_dir: str = None,
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
            token_spans: str = None,
            cpu_only: bool = False,):
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.image_path = image_path
        self.text_prompt = text_prompt
        self.output_dir = output_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.token_spans = token_spans
        self.cpu_only = cpu_only

        self.model = None
        self.image_pil = None
        self.image = None
        self.boxes_filt = None
        self.pred_phrases = None
        self.tgt = None
    
    def load_model(self):
        # load model
        args = SLConfig.fromfile(self.config_file)
        args.device = "cuda" if not self.cpu_only else "cpu"

        model = build_model(args)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)

        model.eval()
        self.model = model

    def load_image(self, image_path=None):
        image_path = image_path or self.image_path

        # load image
        image_pil = Image.open(image_path).convert("RGB")

        transform = T.Compose(
            [
                # T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)  # 3, h, w
        self.image_pil = image_pil
        self.image = image_tensor
        return image_pil, image_tensor

    def demo(self):
        # make output dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.load_model()
        self.load_image()
        self.get_grounding_output()
        image_with_box = self.plot_boxes_to_image()[0]

        image_with_box.save(os.path.join(self.output_dir, "pred.jpg"))

    def _test_scaled(self):
        # make output dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.load_model()
        self.load_image()
        bboxes, labels = self.get_scaled_grounding_output(self.image, self.image_pil)
        
        print("FISHY: ", bboxes)
        print("FISHY: ", labels)


    def get_scaled_grounding_output(self, image, image_pil, text_prompt=None, box_threshold=None, text_threshold=None, with_logits=True):
        self.get_grounding_output(image, text_prompt, box_threshold, text_threshold, with_logits)
        W, H = image_pil.size
        boxes = self.boxes_filt
        labels = self.pred_phrases
        bboxes = []
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]

            x0, y0, x1, y1 = map(int, box)

            bbox = [x0,y0,x1,y1]
            bboxes.append(bbox)
        
        return bboxes, labels

    def get_grounding_output(self, image=None, text_prompt=None, box_threshold=None, text_threshold=None, with_logits=True):
        # Set default arguments from class if not provided
        if image == None:
            image = self.image
        text_prompt = text_prompt or self.text_prompt
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold
        
        assert text_threshold is not None or self.token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        device = "cuda" if not self.cpu_only else "cpu"
        model = self.model.to(device)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # Filter output
        if self.token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # Get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # Build predictions
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # Given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(text_prompt),
                token_span=self.token_spans
            ).to(image.device)  # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(self.token_spans, logits_for_phrases):
                # Get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # Get mask
                filt_mask = logit_phr > box_threshold
                # Filter boxes
                all_boxes.append(boxes[filt_mask])
                # Filter logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases
        
        self.boxes_filt = boxes_filt
        self.pred_phrases = pred_phrases
    
    def _create_tgt(self, image_pil):
        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": self.boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": self.pred_phrases,
        }
        return pred_dict
    
    def plot_boxes_to_image(self, image_pil=None):
        if not image_pil:
            image_pil = self.image_pil
        
        tgt = self._create_tgt(image_pil)

        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())

            x0, y0, x1, y1 = map(int, box)

            # print("Rect FISHY: ", [x0, y0, x1, y1])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

            font = ImageFont.load_default()
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font=font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)

            # print("Rect FISHY 2: ", bbox)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        return image_pil, mask


if __name__ == "__main__":

    text_prompt = "oven" # change for each object
    image_path = "./GroundingDINO/images/oven_ego.png"

    groundingdino = ObjectDINO(image_path=image_path, 
                               text_prompt=text_prompt, 
                               output_dir=output_dir_default)
    
    groundingdino.demo()
    # groundingdino._test_scaled()

    # # make dir
    # os.makedirs(output_dir, exist_ok=True)
    # # load image
    # image_pil, image = load_image(image_path)
    # # load model
    # model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # # set the text_threshold to None if token_spans is set.
    # if token_spans is not None:
    #     text_threshold = None
    #     print("Using token_spans. Set the text_threshold to None.")


    # # run model
    # boxes_filt, pred_phrases = get_grounding_output(
    #     model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
    # )

    # print("BOXES FISH", np.shape(boxes_filt))
    # print("PRED FISH", np.shape(pred_phrases))
    # print("BOXES FISH", boxes_filt)
    # print("PRED FISH", pred_phrases)

    # # visualize pred
    # size = image_pil.size
    # pred_dict = {
    #     "boxes": boxes_filt,
    #     "size": [size[1], size[0]],  # H,W
    #     "labels": pred_phrases,
    # }
    # # import ipdb; ipdb.set_trace()
    # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    # image_with_box.save(os.path.join(output_dir, "pred.jpg"))
