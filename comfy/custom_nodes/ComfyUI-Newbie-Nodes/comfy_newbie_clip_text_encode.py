import sys
import os
import torch
from PIL import Image
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nodes import CLIPTextEncode
from typing import Tuple, Any, Optional


class NewBieCLIPTextEncode(CLIPTextEncode):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "NewBie CLIP model"}),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                    "tooltip": "User prompt for both Gemma and Jina CLIP"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt only for Gemma3-4B-IT (Jina CLIP ignores this)"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image input for Gemma3 multimodal processing"
                }),
                "use_chat_template": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable to use Gemma chat template format (experimental, may differ from training)"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("NewBie CLIP conditioning with dual text encoders",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    TITLE = "NewBie CLIP Text Encode"
    DESCRIPTION = "Encodes prompts using NewBie dual CLIP (Gemma3-4B-IT + Jina CLIP). System prompt only affects Gemma, Jina CLIP only sees user prompt."

    def encode(self, clip, user_prompt: str, system_prompt: str = "", image: Optional[torch.Tensor] = None, use_chat_template: bool = False) -> Tuple[Any,]:
        if not hasattr(clip, 'text_encoder') or not hasattr(clip, 'clip_model'):
            raise ValueError("This node requires a NewBie CLIP model loaded with NewBie CLIP Loader")

        clip._last_text = [user_prompt.strip()] if user_prompt.strip() else [""]

        if image is not None:
            processed_image = self._process_image_for_gemma(image, clip.device)
            result = clip.encode_with_image(user_prompt, processed_image, system_prompt)
            return (result,)
        else:
            # Format prompt for Gemma (with system prompt)
            formatted_prompt = self._format_gemma_chat_template(system_prompt, user_prompt, has_image=False, use_chat_template=use_chat_template)

            # Check if user_prompt has weights (parentheses, brackets, braces)
            import re
            if re.search(r'[()\[\]{}]', user_prompt):
                # User prompt has weights, use encode_text to process them
                # Store the formatted prompt for Gemma
                clip._gemma_prompt = formatted_prompt
                clip._original_prompt = user_prompt
                # Use encode_text which will call tokenize -> encode_from_tokens -> encode_token_weights
                result = clip.encode_text(formatted_prompt)
                return (result,)
            else:
                # No weights, use direct encoding
                if hasattr(clip, '_encode_text_direct'):
                    result = clip._encode_text_direct(formatted_prompt, original_text=user_prompt)
                    return (result,)
                else:
                    # Fallback to standard encode
                    return super().encode(clip, formatted_prompt)
    
    def _process_image_for_gemma(self, image: torch.Tensor, device: str):
        if image.dim() == 4:
            image = image.squeeze(0)
        
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        return pil_image
    
    def _format_gemma_chat_template(self, system_prompt: str, user_prompt: str, has_image: bool = False, use_chat_template: bool = False) -> str:
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        if use_chat_template:
            # Use Gemma chat template format
            combined = ""
            if system_prompt and user_prompt:
                combined = f"{system_prompt}\n\n{user_prompt}"
            elif system_prompt:
                combined = system_prompt
            elif user_prompt:
                combined = user_prompt
            else:
                return ""
            # Apply Gemma3 chat template
            return f"<start_of_turn>user\n{combined}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Match training format: simple concatenation
            if system_prompt and user_prompt:
                if not system_prompt.endswith(" "):
                    system_prompt += " "
                return system_prompt + user_prompt
            elif system_prompt:
                return system_prompt
            elif user_prompt:
                return user_prompt
            else:
                return ""
    
    def _apply_chat_template(self, messages: list, add_generation_prompt: bool = True) -> str:
        if not messages:
            return ""

        result = ""
        system_content = ""

        if messages and messages[0]['role'] == 'system':
            system_content = messages[0]['content']
            messages = messages[1:]

        for i, message in enumerate(messages):
            role = 'model' if message['role'] == 'assistant' else 'user'

            result += f"<start_of_turn>{role}\n"

            if i == 0 and role == 'user' and system_content:
                result += system_content + "\n\n"

            content = message.get('content', '').strip()
            result += content
            result += "<end_of_turn>\n"

        if add_generation_prompt:
            result += "<start_of_turn>model\n"

        return result


class NewBieCLIPTextEncodeBasic(NewBieCLIPTextEncode):
    
    TITLE = "NewBie CLIP Text Encode (Basic)"
    DESCRIPTION = "Basic NewBie CLIP encoding without weight processing. Use for comparison with weight version."
    
    def encode(self, clip, user_prompt: str, system_prompt: str = "", image: Optional[torch.Tensor] = None, use_chat_template: bool = False) -> Tuple[Any,]:
        if not hasattr(clip, 'text_encoder') or not hasattr(clip, 'clip_model'):
            raise ValueError("This node requires a NewBie CLIP model loaded with NewBie CLIP Loader")
        
        clip._last_text = [user_prompt.strip()] if user_prompt.strip() else [""]
        
        if image is not None:
            processed_image = self._process_image_for_gemma(image, clip.device)
            result = clip.encode_with_image(user_prompt, processed_image, system_prompt)
            return (result,)
        else:
            formatted_prompt = self._format_gemma_chat_template(system_prompt, user_prompt, has_image=False, use_chat_template=use_chat_template)
            result = clip._encode_text_direct(formatted_prompt, original_text=user_prompt)
            return (result,)


NODE_CLASS_MAPPINGS = {
    "NewBieCLIPTextEncode": NewBieCLIPTextEncode,
    "NewBieCLIPTextEncodeBasic": NewBieCLIPTextEncodeBasic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieCLIPTextEncode": "NewBie CLIP Text Encode",
    "NewBieCLIPTextEncodeBasic": "NewBie CLIP Text Encode (Basic)",
}