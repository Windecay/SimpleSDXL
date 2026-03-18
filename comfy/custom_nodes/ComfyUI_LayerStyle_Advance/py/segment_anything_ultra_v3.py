# layerstyle advance

from .imagefunc import *
from .segment_anything_func import *


class LS_LoadSAMModels:
    def __init__(self):
        self.NODE_NAME = 'SegmentAnythingUltra V3'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("LS_SAM_MODELS",)
    RETURN_NAMES = ("sam_models", )
    FUNCTION = "load_sam_models"
    CATEGORY = '😺dzNodes/LayerMask'

    def load_sam_models(self, sam_model, grounding_dino_model):


        SAM_MODEL = load_sam_model(sam_model)
        DINO_MODEL = load_groundingdino_model(grounding_dino_model)

        return ({"SAM_MODEL":SAM_MODEL, "DINO_MODEL":DINO_MODEL},)


class LS_SegmentAnythingUltraV3:
    def __init__(self):
        self.NODE_NAME = 'SegmentAnythingUltra V3'

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'vitmatte-base-composition-1k', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_models": ("LS_SAM_MODELS", ),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject"}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segment_anything_ultra_v3"
    CATEGORY = '😺dzNodes/LayerMask'

    def segment_anything_ultra_v3(self, image, sam_models, threshold,
                                  detail_method, detail_erode, detail_dilate,
                                  black_point, white_point, process_detail, prompt,
                                  device, max_megapixels,
                                  ):

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False



        SAM_MODEL = sam_models["SAM_MODEL"]
        DINO_MODEL = sam_models["DINO_MODEL"]

        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            _image = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(DINO_MODEL, _image, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(SAM_MODEL, _image, boxes)
            _mask = _mask[0]
            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels, method=detail_method)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)
            _image = RGB2RGBA(tensor2pil(i).convert('RGB'), _mask.convert('L'))

            ret_images.append(pil2tensor(_image))
            ret_masks.append(image2mask(_mask))
        if len(ret_masks) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)


        log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra V3": LS_SegmentAnythingUltraV3,
    "LayerMask: LoadSegmentAnythingModels": LS_LoadSAMModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegmentAnythingUltra V3": "LayerMask: SegmentAnythingUltra V3(Advance)",
    "LayerMask: LoadSegmentAnythingModels": "LayerMask: Load SegmentAnything Models(Advance)",
}
