# layerstyle advance

import io
from .imagefunc import *

def is_only_digits_and_spaces(s:str) -> bool:
    return bool(re.fullmatch(r'[0-9\s]*', s))

class LS_GeminiNode:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "run_gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'Gemini'

    @classmethod
    def INPUT_TYPES(self):
        gemini_model_list = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-exp",
            "learnlm-1.5-pro-experimental"]
        language_list = ['en', 'zh-CN']
        return {
            "required": {
                "model": (gemini_model_list,),
                "max_output_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 2, "step": 0.1}),
                "words_limit": ("INT", {"default": 200, "min": 8, "max": 2048, "step": 1}),
                "response_language": (language_list,),
                "system_prompt": ("STRING",
                                  {"default": "You are creating a prompt for Stable Diffusion to generate an image.",
                                   "multiline": False}),
                "user_prompt": ("STRING", {
                    "default": "Generate a prompt about a girl.",
                    "multiline": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    def run_gemini(self, model, system_prompt, user_prompt, max_output_tokens, temperature,
                   words_limit, response_language, image_1=None, image_2=None):

        import google.generativeai as genai
        ret_texts = []

        g_model = genai.GenerativeModel(model,
                                      generation_config=gemini_generate_config,
                                      safety_settings=gemini_safety_settings)
        g_cfg = genai.GenerationConfig(temperature=temperature,
                                       max_output_tokens=max_output_tokens)
        genai.configure(api_key=get_api_key('google_api_key'), transport='rest')
        prompt = {
                  "USER_INPUT":user_prompt,
                  "action": f"{system_prompt}\n"
                            f"Follow the USER_INPUT to complete task, keep response length between {int(words_limit * 0.8)} to {int(words_limit * 1.2)} words.",
                  "output_format": {
                    "content": f"Only return the final result, not include any unnecessary content.",
                    "language": response_language,
                  }
                }
        prompt = json.dumps(prompt)
        log(f"{self.NODE_NAME}: Request to {model}...")
        if image_1 is not None and image_2 is not None:
            for index,img in enumerate(image_1):
                _image1 = tensor2pil(img.unsqueeze(0)).convert('RGB')
                _image2 = tensor2pil(image_2[index].unsqueeze(0)).convert('RGB') if index < len(image_2) else tensor2pil(image_2[-1].unsqueeze(0)).convert('RGB')
                response = g_model.generate_content([prompt, _image1, _image2], generation_config=g_cfg)
                ret_text = response.text
                log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
                ret_texts.append(ret_text)
        elif (image_1 is not None and image_2 is None) or (image_2 is not None and image_1 is None):
            _imgs = image_1 if image_1 is not None else image_2
            for img in _imgs:
                _image = tensor2pil(img.unsqueeze(0)).convert('RGB')
                response = g_model.generate_content([prompt, _image], generation_config=g_cfg)
                ret_text = response.text
                log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
                ret_texts.append(ret_text)
        else:
            response = g_model.generate_content(prompt, generation_config=g_cfg)
            ret_text = response.text
            log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
            ret_texts.append(ret_text)

        return (ret_texts,)

class LS_GeminiNode_V2:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "run_gemini_v2"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'GeminiV2'

    @classmethod
    def INPUT_TYPES(self):
        gemini_model_list = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-pro-exp-03-25",
        ]
        language_list = ['en', 'zh-CN']
        return {
            "required": {
                "model": (gemini_model_list,),
                "max_output_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 2, "step": 0.1}),
                "words_limit": ("INT", {"default": 200, "min": 8, "max": 2048, "step": 1}),
                "response_language": (language_list,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "system_prompt": ("STRING",
                                  {"default": "You are creating a prompt for Stable Diffusion to generate an image.",
                                   "multiline": False}),
                "user_prompt": ("STRING", {
                    "default": "Generate a prompt about a girl.",
                    "multiline": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    def run_gemini_v2(self, model, system_prompt, user_prompt, max_output_tokens, temperature,
                   words_limit, response_language, seed, image_1=None, image_2=None):


        from google import genai
        from google.genai import types
        ret_texts = []


        client = genai.Client(api_key=get_api_key('google_api_key'))

        gen_config = types.GenerateContentConfig(
            safety_settings=gemini_safety_settings,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed
        )

        contents = []
        prompt_text = f"USER_INPUT: {user_prompt}\n" \
                      f"{system_prompt}\n" \
                      f"Follow the USER_INPUT to complete task, Only output the positive prompt, keep response length between {int(words_limit * 0.8)} to {int(words_limit * 1.2)} words."

        contents.append({"text": prompt_text})
        log(f"{self.NODE_NAME}: Request to {model}...")
        if image_1 is not None and image_2 is not None: # 2张图
            for index,img in enumerate(image_1):
                images = []
                _image1 = tensor2pil(img.unsqueeze(0)).convert('RGB')
                images.append(_image1)
                _image2 = tensor2pil(image_2[index].unsqueeze(0)).convert('RGB') if index < len(image_2) else tensor2pil(image_2[-1].unsqueeze(0)).convert('RGB')
                images.append(_image2)
                for i in images:
                    img_byte_arr = io.BytesIO()
                    i.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_bytes = img_byte_arr.read()
                    img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                    contents.append(img_part)
                    contents[0]["text"] += "\nUse these reference images as guidance."

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config
                )
                ret_text = response.text
                log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
                ret_texts.append(ret_text)
        elif (image_1 is not None and image_2 is None) or (image_2 is not None and image_1 is None): # 1张图
            _imgs = image_1 if image_1 is not None else image_2
            for img in _imgs:
                _image = tensor2pil(img.unsqueeze(0)).convert('RGB')

                img_byte_arr = io.BytesIO()
                _image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes = img_byte_arr.read()
                img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                contents.append(img_part)
                contents[0]["text"] += "\nUse this reference image as guidance."

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config
                )
                ret_text = response.text
                log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
                ret_texts.append(ret_text)
        else: # 无图
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )
            ret_text = response.text
            log(f"{self.NODE_NAME}: Gemini response is:\n\033[1;36m{ret_text}\033[m")
            ret_texts.append(ret_text)

        return (ret_texts,)


class LS_OBJECT_DETECTOR_Gemini:

    CATEGORY = '😺dzNodes/LayerMask'
    FUNCTION = "run_gemini_detect"
    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    # OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'GeminiDetect'

    @classmethod
    def INPUT_TYPES(self):
        gemini_model_list = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-exp"
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (gemini_model_list,),
                "prompt": ("STRING", {"default": "subject"}),
            },
            "optional": {
            }
        }

    def run_gemini_detect(self, image, model, prompt):

        import google.generativeai as genai
        ret_bboxes = []
        ret_previews = []

        g_model = genai.GenerativeModel(model,
                                      generation_config=gemini_generate_config,
                                      safety_settings=gemini_safety_settings)

        genai.configure(api_key=get_api_key('google_api_key'), transport='rest')
        g_prompt = f"Return a bounding box of {prompt} in this image in [ymin, xmin, ymax, xmax] format. Only return these 4 values, separated by a space, without any extra characters."

        log(f"{self.NODE_NAME}: Request to {model}...")

        for img in image:
            _image = tensor2pil(img.unsqueeze(0)).convert('RGB')
            response = g_model.generate_content([_image, g_prompt])
            ret_text = response.text

            if not is_only_digits_and_spaces(ret_text):
                ret_bboxes.append([(-1, -1, 0, 0)])
                ret_previews.append(pil2tensor(_image))
                log(f"{self.NODE_NAME} no object found", message_type='warning')
                continue

            y1,x1,y2,x2 = [int(x) for x in ret_text.split()]

            # Convert normalized coordinates to absolute coordinates
            x1 = int(x1 / 1000 * _image.width)
            y1 = int(y1 / 1000 * _image.height)
            x2 = int(x2 / 1000 * _image.width)
            y2 = int(y2 / 1000 * _image.height)
            bboxes = standardize_bbox([(x1, y1, x2, y2)])
            preview = draw_bounding_boxes(_image.convert("RGB"), bboxes, color="random", line_width=-1)
            ret_previews.append(pil2tensor(preview))

            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')
            ret_bboxes.append(bboxes)

        return (ret_bboxes, torch.cat(ret_previews, dim=0),)

class LS_OBJECT_DETECTOR_Gemini_V2:

    CATEGORY = '😺dzNodes/LayerMask'
    FUNCTION = "run_gemini_detect_v2"
    RETURN_TYPES = ("BBOXES", "IMAGE",)
    RETURN_NAMES = ("bboxes", "preview",)
    # OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'GeminiDetectV2'

    @classmethod
    def INPUT_TYPES(self):
        gemini_model_list = [
            "gemini-2.5-pro-exp-03-25",
            "gemini-1.5-pro",
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (gemini_model_list,),
                "prompt": ("STRING", {"default": "subject"}),
            },
            "optional": {
            }
        }

    def run_gemini_detect_v2(self, image, model, prompt):

        from google import genai
        from google.genai import types
        ret_bboxes = []
        ret_previews = []

        client = genai.Client(api_key=get_api_key('google_api_key'))

        gen_config = types.GenerateContentConfig(
            safety_settings=gemini_safety_settings,
        )

        contents = []
        prompt_text = f"Return a bounding box of {prompt} in this image in [ymin, xmin, ymax, xmax] format. Only return these 4 values, separated by a space, without any extra characters."
        contents.append({"text": prompt_text})

        log(f"{self.NODE_NAME}: Request to {model}...")

        for img in image:
            _image = tensor2pil(img.unsqueeze(0)).convert('RGB')

            img_byte_arr = io.BytesIO()
            _image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            image_bytes = img_byte_arr.read()
            img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
            contents.append(img_part)
            # contents[0]["text"] += f"\nUse this image as find {prompt}'s bounding box."

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            ret_text = response.text

            if not is_only_digits_and_spaces(ret_text):
                ret_bboxes.append([(-1, -1, 0, 0)])
                ret_previews.append(pil2tensor(_image))
                log(f"{self.NODE_NAME} no object found", message_type='warning')
                continue

            y1,x1,y2,x2 = [int(x) for x in ret_text.split()]

            # Convert normalized coordinates to absolute coordinates
            x1 = int(x1 / 1000 * _image.width)
            y1 = int(y1 / 1000 * _image.height)
            x2 = int(x2 / 1000 * _image.width)
            y2 = int(y2 / 1000 * _image.height)
            bboxes = standardize_bbox([(x1, y1, x2, y2)])
            preview = draw_bounding_boxes(_image.convert("RGB"), bboxes, color="random", line_width=-1)
            ret_previews.append(pil2tensor(preview))

            log(f"{self.NODE_NAME} found {len(bboxes)} object(s)", message_type='info')
            ret_bboxes.append(bboxes)

        return (ret_bboxes, torch.cat(ret_previews, dim=0),)


class LS_Gemini_Image_Edit:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "run_gemini_image_edit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)


    def __init__(self):
        self.NODE_NAME = 'GeminiImageEdit'

    @classmethod
    def INPUT_TYPES(self):
        gemini_model_list = [
            "gemini-2.0-flash-exp-image-generation",
            "gemini-2.5-flash-image-preview"
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (gemini_model_list,),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 2, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "user_prompt": ("STRING", {
                    "default": "change the background to forest",
                    "multiline": True}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    def run_gemini_image_edit(self, image, model, temperature, seed, user_prompt,
                              image_2=None, image_3=None):


        from google import genai
        from google.genai import types
        ret_images = []


        client = genai.Client(api_key=get_api_key('google_api_key'))
        gen_config = types.GenerateContentConfig(
            safety_settings=gemini_safety_settings,
            temperature=temperature,
            seed=seed,
            response_modalities=['Text', 'Image']
        )




        log(f"{self.NODE_NAME}: Request to {model}...")

        for idx,img in enumerate(image):
            input_images = []
            input_images.append(tensor2pil(img.unsqueeze(0)).convert('RGB'))

            contents = []
            prompt_text = f"Create a detailed image of: {user_prompt}."
            contents.append({"text": prompt_text})
            if image_2 is not None:
                img2 = tensor2pil(image_2[idx].unsqueeze(0)).convert('RGB') if idx < len(image_2) else tensor2pil(image_2[-1].unsqueeze(0)).convert('RGB')
                input_images.append(img2)
            if image_3 is not None:
                img3 = tensor2pil(image_3[idx].unsqueeze(0)).convert('RGB') if idx < len(image_3) else tensor2pil(image_3[-1].unsqueeze(0)).convert('RGB')
                input_images.append(img3)

            for i in input_images:
                img_byte_arr = io.BytesIO()
                i.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes = img_byte_arr.read()
                img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                contents.append(img_part)
            if len(input_images) > 1:
                contents[0]["text"] += f"\nBased on the first image, Use other reference image as guidance."
            else:
                contents[0]["text"] += f"\nBased on this image."

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            for item in response.candidates[0].content.parts:
                if hasattr(item, "inline_data") and item.inline_data is not None and item.inline_data.mime_type == "image/png":
                    image_bytes = item.inline_data.data
                    image_bytes = io.BytesIO(image_bytes)
                    image_bytes.seek(0)
                    ret_images.append(pil2tensor(Image.open(image_bytes)))

        if len(ret_images) == 0:
            log(f"{self.NODE_NAME} no image response, return original image", message_type='warning')
            ret_images=image

        return (ret_images)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: Gemini": LS_GeminiNode,
    "LayerUtility: GeminiV2": LS_GeminiNode_V2,
    "LayerMask: ObjectDetectorGemini": LS_OBJECT_DETECTOR_Gemini,
    "LayerMask: ObjectDetectorGeminiV2": LS_OBJECT_DETECTOR_Gemini_V2,
    "LayerUtility: GeminiImageEdit": LS_Gemini_Image_Edit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: Gemini": "LayerUtility: Gemini(Advance)",
    "LayerUtility: GeminiV2": "LayerUtility: Gemini V2(Advance)",
    "LayerMask: ObjectDetectorGemini": "LayerMask: Object Detector Gemini(Advance)",
    "LayerMask: ObjectDetectorGeminiV2": "LayerMask: Object Detector Gemini V2(Advance)",
    "LayerUtility: GeminiImageEdit": "LayerUtility: Gemini Image Edit(Advance)",
}
