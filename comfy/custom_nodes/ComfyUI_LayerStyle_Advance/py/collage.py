import torch
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import copy
from .imagefunc import log, tensor2pil, pil2tensor, image2mask
from .imagefunc import fit_resize_image, extract_numbers, gaussian_blur, mask_area, draw_rounded_rectangle

class LS_CollageGenerator:
    """
    随机分割生成指定数量的不规则小矩形。
    """
    def __init__(self, width, height, num, border_width, r, uniformity, seed ):
        self.width = width
        self.height = height
        self.num = num
        self.border_width = int((self.width + self.height) * border_width / 200)
        self.r = r
        self.seed = seed
        self.split_num = int(1e18)
        self.uniformity = uniformity
        self.rectangles = self.adjust_bboxes_with_gaps(self.split_rec())

    def split_rec(self):
        random.seed(self.seed)
        if self.num <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Value mast be positive integer")

        current_rectangles = [(0, 0, self.width, self.height, 0)]

        while len(current_rectangles) < self.num:
            split_counts = [rect[4] for rect in current_rectangles]
            min_splits = min(split_counts)
            max_splits = max(split_counts)
            probabilities = []

            for rect in current_rectangles:
                split_count = rect[4]
                normalized_splits = (split_count - min_splits) / (
                    max_splits - min_splits if max_splits > min_splits else 1)
                probability = 1 - (normalized_splits * (1 - self.uniformity))
                probabilities.append(probability)
            if sum(probabilities) > 0:
                probabilities = [p / sum(probabilities) for p in probabilities]
            else:
                probabilities = [1.0 / len(probabilities)] * len(probabilities)

            rect_index = random.choices(range(len(current_rectangles)),
                                        weights=probabilities, k=1)[0]

            x, y, w, h, split_count = current_rectangles.pop(rect_index)

            if w > h or (w == h and random.choice([True, False])):
                split = random.uniform(0.3, 0.7) * w
                rect1 = (x, y, split, h, split_count + 1)
                rect2 = (x + split, y, w - split, h, split_count + 1)
            else:
                split = random.uniform(0.3, 0.7) * h
                rect1 = (x, y, w, split, split_count + 1)
                rect2 = (x, y + split, w, h - split, split_count + 1)

            current_rectangles.extend([rect1, rect2])

        rectangles = [(int(x), int(y), int(w), int(h))
                      for x, y, w, h, _ in current_rectangles]

        return rectangles

    def adjust_bboxes_with_gaps(self, rectangles):
        MIN_SIZE = 1
        adjusted_bboxes = []

        for x, y, w, h in rectangles:
            new_x = min(x + self.border_width, self.width - MIN_SIZE)
            new_y = min(y + self.border_width, self.height - MIN_SIZE)
            new_w = max(MIN_SIZE, w - 2 * self.border_width)
            new_h = max(MIN_SIZE, h - 2 * self.border_width)

            if new_x + new_w > self.width:
                new_x = max(0, self.width - new_w)
            if new_y + new_h > self.height:
                new_y = max(0, self.height - new_h)

            adjusted_bboxes.append((new_x, new_y, new_w, new_h))

        return adjusted_bboxes

    def draw_mask(self):
        bboxes = []

        for bbox in self.rectangles:
            bboxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        scale_factor = 2

        img = Image.new('RGB', (self.width, self.height), color='white')
        img = draw_rounded_rectangle(img, self.r, bboxes, scale_factor, color='black')

        return img


class LS_Collage:
    def __init__(self):
        self.NODE_NAME = 'Collage'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "canvas_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 16}),
                "canvas_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 16}),
                "border_width": ("FLOAT", {"default": 2, "min": 0, "max": 20, "step": 0.1}),
                "rounded_rect_radius": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "uniformity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1}), # 分割均匀权重 0均匀分割，1不均匀分割
                "background_color": ("STRING", {"default": "#FFFFFF"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),
            },
            "optional": {
                "florence2_model": ("FLORENCE2",),
                "object_prompt": ("STRING", {"default": "face"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "collage"
    CATEGORY = '😺dzNodes/LayerUtility'

    def collage(self, images, canvas_width, canvas_height, border_width, rounded_rect_radius,
                   uniformity, background_color, seed, florence2_model=None, object_prompt="face"):

        batch_size = images.shape[0]

        rects = LS_CollageGenerator(width=canvas_width,
                           height=canvas_height,
                           num=batch_size,
                           border_width=border_width,
                           r=rounded_rect_radius,
                           uniformity=uniformity,
                           seed=seed)

        rects_border_image = rects.draw_mask()
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=background_color)
        color_image = copy.deepcopy(canvas)
        from .object_detector import LS_OBJECT_DETECTOR_FL2

        for i in tqdm(range(batch_size)):
            img = tensor2pil(images[i]).convert("RGB")
            img_x = rects.rectangles[i][0]
            img_y = rects.rectangles[i][1]
            img_target_width = rects.rectangles[i][2]
            img_target_height = rects.rectangles[i][3]

            od = LS_OBJECT_DETECTOR_FL2()
            if florence2_model is not None:
                bboxes = od.object_detector_fl2(image=[images[i]], prompt=object_prompt, florence2_model=florence2_model,
                                                sort_method="confidence", bbox_select="first", select_index="0")[0]
                bbox_mask = self.draw_bbox_mask(img, bboxes, 0, 0, 0, 0)
                resized_img = self.image_auto_crop_v3(img, img_target_width, img_target_height, bbox_mask)
            else:
                resized_img = fit_resize_image(img, img_target_width, img_target_height, fit="crop", resize_sampler=Image.LANCZOS)

            canvas.paste(resized_img, box=(img_x, img_y))
        canvas.paste(color_image, box=(0, 0), mask=rects_border_image.convert("L"))

        return (pil2tensor(canvas), 1 - image2mask(rects_border_image),)

    def draw_bbox_mask(self, image, bboxes, grow_top, grow_bottom, grow_left, grow_right
                      ):

        mask = Image.new("L", image.size, color='black')
        for bbox in bboxes:
            try:
                if len(bbox) == 0:
                    continue
                else:
                    x1, y1, x2, y2 = bbox
            except ValueError:
                if len(bbox) == 0:
                    continue
                else:
                    x1, y1, x2, y2 = bbox[0]
            w = x2 - x1
            h = y2 - y1
            if grow_top:
                y1 = int(y1 - h * grow_top)
            if grow_bottom:
                y2 = int(y2 + h * grow_bottom)
            if grow_left:
                x1 = int(x1 - w * grow_left)
            if grow_right:
                x2 = int(x2 + w * grow_right)
            if y1 > y2 or x1 > x2:
                continue
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill='white', outline='white', width=0)

        return mask

    def image_auto_crop_v3(self, image, proportional_width, proportional_height, mask,
                        ):

        scale_to_length = proportional_width
        _image = image
        ratio = proportional_width / proportional_height
        resize_sampler = Image.LANCZOS
        # calculate target width and height
        if ratio > 1:
            target_width = scale_to_length
            target_height = int(target_width / ratio)
        else:
            target_width = scale_to_length
            target_height = int(target_width / ratio)

        _mask = mask
        bluredmask = gaussian_blur(_mask, 20).convert('L')
        (mask_x, mask_y, mask_w, mask_h) = mask_area(bluredmask)
        orig_ratio = _image.width / _image.height
        target_ratio = target_width / target_height
        # crop image to target ratio
        if orig_ratio > target_ratio: # crop LiftRight side
            crop_w = int(_image.height * target_ratio)
            crop_h = _image.height
        else: # crop TopBottom side
            crop_w = _image.width
            crop_h = int(_image.width / target_ratio)
        crop_x = mask_w // 2 + mask_x - crop_w // 2
        if crop_x < 0:
            crop_x = 0
        if crop_x + crop_w > _image.width:
            crop_x = _image.width - crop_w
        crop_y = mask_h // 2 + mask_y - crop_h // 2
        if crop_y < 0:
            crop_y = 0
        if crop_y + crop_h > _image.height:
            crop_y = _image.height - crop_h
        crop_image = _image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        ret_image = crop_image.resize((target_width, target_height), resize_sampler)

        return ret_image


NODE_CLASS_MAPPINGS = {
    "LayerUtility: Collage": LS_Collage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: Collage": "LayerUtility: Collage(Advance)",
}