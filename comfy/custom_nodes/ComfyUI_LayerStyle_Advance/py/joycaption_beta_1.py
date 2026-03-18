import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
import folder_paths
from comfy.utils import ProgressBar
from .imagefunc import download_hg_model, log, tensor2pil, clear_memory


# caption type
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."

QUANTIZATION_CONFIGS = {
    "nf4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        },
    "int8": {
        "load_in_8bit": True,
    },
    "bf16": {},
}

class JoyCaptionPredictor:
    def __init__(self, model: str, quantization_mode: str, device:str):
        checkpoint_path = download_hg_model(model, "LLavacheckpoints")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)

        if quantization_mode == "bf16":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype="bfloat16",
                device_map="auto"
            )
        else:
            from transformers import BitsAndBytesConfig
            qnt_config = BitsAndBytesConfig(
                **QUANTIZATION_CONFIGS[quantization_mode],
                # llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                # Transformer's Siglip implementation has bugs when quantized, so skip those.
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=qnt_config
            )

        self.model.eval()

    @torch.inference_mode()
    def generate(self, image:Image, system:str, prompt:str, max_new_tokens:int, temperature:float,
                 top_p:float, top_k:int) -> str:
        convo = [
            {
                "role": "system",
                "content": system.strip(),
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]

        # Format the conversation
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            suppress_tokens=None,
            use_cache=True,
            temperature=temperature,
            top_k=None if top_k == 0 else top_k,
            top_p=top_p,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        return caption.strip()


class LS_JoyCaptionBetaExtraOptions:

    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "joycaptionbeta_extra_choice"
    RETURN_TYPES = ("JoyCaption2ExtraOption",)
    RETURN_NAMES = ("extra_option",)

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "refer_character_name": ("BOOLEAN", {"default": False}),
                "exclude_people_info": ("BOOLEAN", {"default": False}),
                "include_lighting": ("BOOLEAN", {"default": False}),
                "include_camera_angle": ("BOOLEAN", {"default": False}),
                "include_watermark": ("BOOLEAN", {"default": False}),
                "include_JPEG_artifacts": ("BOOLEAN", {"default": False}),
                "include_exif": ("BOOLEAN", {"default": False}),
                "exclude_sexual": ("BOOLEAN", {"default": False}),
                "exclude_image_resolution": ("BOOLEAN", {"default": False}),
                "include_aesthetic_quality": ("BOOLEAN", {"default": False}),
                "include_composition_style": ("BOOLEAN", {"default": False}),
                "exclude_text": ("BOOLEAN", {"default": False}),
                "specify_depth_field": ("BOOLEAN", {"default": False}),
                "specify_lighting_sources": ("BOOLEAN", {"default": False}),
                "do_not_use_ambiguous_language": ("BOOLEAN", {"default": False}),
                "include_nsfw": ("BOOLEAN", {"default": False}),
                "only_describe_most_important_elements": ("BOOLEAN", {"default": False}),
                "do_not_include_artist_name_or_title": ("BOOLEAN", {"default": False}),
                "identify_image_orientation": ("BOOLEAN", {"default": False}),
                "use_vulgar_slang_and_profanity": ("BOOLEAN", {"default": False}),
                "do_not_use_polite_euphemisms": ("BOOLEAN", {"default": False}),
                "include_character_age": ("BOOLEAN", {"default": False}),
                "include_camera_shot_type": ("BOOLEAN", {"default": False}),
                "exclude_mood_feeling": ("BOOLEAN", {"default": False}),
                "include_camera_vantage_height": ("BOOLEAN", {"default": False}),
                "mention_watermark": ("BOOLEAN", {"default": False}),
                "avoid_meta_descriptive_phrases": ("BOOLEAN", {"default": False}),
                "character_name": ("STRING", {"default": "Huluwa", "multiline": False}),
            },
            "optional": {
            }
        }

    def joycaptionbeta_extra_choice(self, refer_character_name, exclude_people_info, include_lighting, include_camera_angle,
                     include_watermark, include_JPEG_artifacts, include_exif, exclude_sexual,
                     exclude_image_resolution, include_aesthetic_quality, include_composition_style,
                     exclude_text, specify_depth_field, specify_lighting_sources,
                     do_not_use_ambiguous_language, include_nsfw, only_describe_most_important_elements,
                     do_not_include_artist_name_or_title, identify_image_orientation, use_vulgar_slang_and_profanity,
                     do_not_use_polite_euphemisms, include_character_age, include_camera_shot_type,
                     exclude_mood_feeling, include_camera_vantage_height, mention_watermark, avoid_meta_descriptive_phrases,
                     character_name):

        extra_list = {
            "refer_character_name":"If there is a person/character in the image you must refer to them as {name}.",
            "exclude_people_info":"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
            "include_lighting":"Include information about lighting.",
            "include_camera_angle":"Include information about camera angle.",
            "include_watermark":"Include information about whether there is a watermark or not.",
            "include_JPEG_artifacts":"Include information about whether there are JPEG artifacts or not.",
            "include_exif":"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
            "exclude_sexual":"Do NOT include anything sexual; keep it PG.",
            "exclude_image_resolution":"Do NOT mention the image's resolution.",
            "include_aesthetic_quality":"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
            "include_composition_style":"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
            "exclude_text":"Do NOT mention any text that is in the image.",
            "specify_depth_field":"Specify the depth of field and whether the background is in focus or blurred.",
            "specify_lighting_sources":"If applicable, mention the likely use of artificial or natural lighting sources.",
            "do_not_use_ambiguous_language":"Do NOT use any ambiguous language.",
            "include_nsfw":"Include whether the image is sfw, suggestive, or nsfw.",
            "only_describe_most_important_elements":"ONLY describe the most important elements of the image.",
            "do_not_include_artist_name_or_title":"If it is a work of art, do not include the artist's name or the title of the work.",
            "identify_image_orientation":"Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
            "use_vulgar_slang_and_profanity":"""Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
            "do_not_use_polite_euphemisms":"Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
            "include_character_age":"Include information about the ages of any people/characters when applicable.",
            "include_camera_shot_type":"Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
            "exclude_mood_feeling":"Do not mention the mood/feeling/etc of the image.",
            "include_camera_vantage_height":"Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
            "mention_watermark":"If there is a watermark, you must mention it.",
            "avoid_meta_descriptive_phrases":""""Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.""",
        }
        ret_list = []
        if refer_character_name:
            ret_list.append(extra_list["refer_character_name"])
        if exclude_people_info:
            ret_list.append(extra_list["exclude_people_info"])
        if include_lighting:
            ret_list.append(extra_list["include_lighting"])
        if include_camera_angle:
            ret_list.append(extra_list["include_camera_angle"])
        if include_watermark:
            ret_list.append(extra_list["include_watermark"])
        if include_JPEG_artifacts:
            ret_list.append(extra_list["include_JPEG_artifacts"])
        if include_exif:
            ret_list.append(extra_list["include_exif"])
        if exclude_sexual:
            ret_list.append(extra_list["exclude_sexual"])
        if exclude_image_resolution:
            ret_list.append(extra_list["exclude_image_resolution"])
        if include_aesthetic_quality:
            ret_list.append(extra_list["include_aesthetic_quality"])
        if include_composition_style:
            ret_list.append(extra_list["include_composition_style"])
        if exclude_text:
            ret_list.append(extra_list["exclude_text"])
        if specify_depth_field:
            ret_list.append(extra_list["specify_depth_field"])
        if specify_lighting_sources:
            ret_list.append(extra_list["specify_lighting_sources"])
        if do_not_use_ambiguous_language:
            ret_list.append(extra_list["do_not_use_ambiguous_language"])
        if include_nsfw:
            ret_list.append(extra_list["include_nsfw"])
        if only_describe_most_important_elements:
            ret_list.append(extra_list["only_describe_most_important_elements"])
        if do_not_include_artist_name_or_title:
            ret_list.append(extra_list["do_not_include_artist_name_or_title"])
        if identify_image_orientation:
            ret_list.append(extra_list["identify_image_orientation"])
        if use_vulgar_slang_and_profanity:
            ret_list.append(extra_list["use_vulgar_slang_and_profanity"])
        if do_not_use_polite_euphemisms:
            ret_list.append(extra_list["do_not_use_polite_euphemisms"])
        if include_character_age:
            ret_list.append(extra_list["include_character_age"])
        if include_camera_shot_type:
            ret_list.append(extra_list["include_camera_shot_type"])
        if exclude_mood_feeling:
            ret_list.append(extra_list["exclude_mood_feeling"])
        if include_camera_vantage_height:
            ret_list.append(extra_list["include_camera_vantage_height"])
        if mention_watermark:
            ret_list.append(extra_list["mention_watermark"])
        if avoid_meta_descriptive_phrases:
            ret_list.append(extra_list["avoid_meta_descriptive_phrases"])

        return ([ret_list, character_name],)


class LS_LoadJoyCaptionB1Model:
    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "load_joycaptionB1_model"
    RETURN_TYPES = ("JOYCAPTIONBETA1_MODEL",)
    RETURN_NAMES = ("joycaption_beta1_model",)

    def __init__(self):
        self.NODE_NAME = 'Load JoyCaptionBetaOne Model'

    @classmethod
    def INPUT_TYPES(self):
        model_list = ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
        quantization_mode_list = list(QUANTIZATION_CONFIGS.keys())
        device_list = ["cuda", "cpu"]
        return {
            "required": {
                "model": (model_list,),
                "quantization_mode": (quantization_mode_list, {"default": "nf4"}),
                "device": (device_list, {"default": "cuda"},),
            },
            "optional": {
            }
        }

    def load_joycaptionB1_model(self, model, quantization_mode, device):
        log(f"{self.NODE_NAME}: Loaded model {model} with {quantization_mode}")
        return (JoyCaptionPredictor(model, quantization_mode, device),)


class LS_JoyCaptionB1:
    CATEGORY = '😺dzNodes/LayerUtility'
    FUNCTION = "joycaptionB1"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'JoyCaptionBetaOne'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(self):

        caption_type_list = list(CAPTION_TYPE_MAP.keys())
        caption_length_list = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261,10)]

        return {
            "required": {
                "image": ("IMAGE",),
                "joycaption_beta1_model": ("JOYCAPTIONBETA1_MODEL",),
                "caption_type": (caption_type_list,),
                "caption_length": (caption_length_list,),
                "max_new_tokens": ("INT", {"default": 512, "min": 4, "max": 2048, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 2, "step": 0.01}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "extra_options": ("JoyCaption2ExtraOption",),
            }
        }

    def build_prompt(self, caption_type: str, caption_length: str | int, extra_options: list[str] = None,
                     user_prompt: str = "") -> str:

        # 如果提供了user_prompt，直接使用它
        if user_prompt and user_prompt.strip():
            prompt = user_prompt.strip()
        else:
            # 选择正确的模板行
            if caption_length == "any":
                map_idx = 0
            elif isinstance(caption_length, str) and caption_length.isdigit():
                map_idx = 1  # 数字字数模板
            else:
                map_idx = 2  # 长度描述符模板

            prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

        if extra_options is not None:
            extra, name_input = extra_options
            prompt += " " + " ".join(extra)
        else:
            name_input = "{NAME}"

        return prompt.format(
            name=name_input,
            length=caption_length,
            word_count=caption_length,
        )

    def joycaptionB1(self, image, joycaption_beta1_model, caption_type, caption_length,
                     max_new_tokens, top_p, top_k, temperature, user_prompt,
                     extra_options=None):

        ret_text = []
        predictor = joycaption_beta1_model

        # 构建提示词
        system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
        prompt = self.build_prompt(caption_type, caption_length, extra_options, user_prompt)

        comfy_pbar = ProgressBar(image.shape[0])
        tqdm_pbar = tqdm(total=image.shape[0], desc="Processing Images")
        for img in image:
            img = tensor2pil(img.unsqueeze(0)).convert('RGB')
            caption = predictor.generate(
                image=img,
                system=system_prompt,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            log(f"{self.NODE_NAME}: caption={caption}")
            ret_text.append(caption)
            comfy_pbar.update(1)
            tqdm_pbar.update(1)

        return (ret_text,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: JoyCaptionBeta1": LS_JoyCaptionB1,
    "LayerUtility: LoadJoyCaptionBeta1Model": LS_LoadJoyCaptionB1Model,
    "LayerUtility: JoyCaptionBeta1ExtraOptions": LS_JoyCaptionBetaExtraOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: JoyCaptionBeta1": "LayerUtility: JoyCaption Beta One (Advance)",
    "LayerUtility: LoadJoyCaptionBeta1Model": "LayerUtility: Load JoyCaption Beta One Model (Advance)",
    "LayerUtility: JoyCaptionBeta1ExtraOptions": "LayerUtility: JoyCaption Beta One Extra Options(Advance)"
} 