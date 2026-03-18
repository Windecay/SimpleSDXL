# layer style advance
import torch
from .imagefunc import log, tensor2pil, gaussian_blur, get_image_color_tone, get_image_color_average, RGB_to_HSV, Hex_to_RGB

class GetColorTone:

    def __init__(self):
        self.NODE_NAME = 'GetColorTone'

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['main_color', 'average']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (mode_list,),  # 主色/平均色
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("RGB color in HEX", "HSV color in list")
    FUNCTION = 'get_color_tone'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_color_tone(self, image, mode,):
        if image.shape[0] > 0:
            image = torch.unsqueeze(image[0], 0)
        _canvas = tensor2pil(image).convert('RGB')
        _canvas = gaussian_blur(_canvas, int((_canvas.width + _canvas.height) / 200))
        if mode == 'main_color':
            ret_color = get_image_color_tone(_canvas)
        else:
            ret_color = get_image_color_average(_canvas)
        hsv_color = RGB_to_HSV(Hex_to_RGB(ret_color))
        log(f"{self.NODE_NAME}: color is {ret_color}/{hsv_color}", message_type='finish')
        return (ret_color, hsv_color)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetColorTone": GetColorTone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetColorTone": "LayerUtility: GetColorTone(Advance)"
}