# layerstyle advance
import os
import torch
import re
import folder_paths
from PIL import Image
from .imagefunc import log, check_and_download_model, tensor2pil


smollm2_repo = {
    "SmolLM2-135M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}

smolvlm_repo = {
    "SmolVLM-Instruct": "HuggingFaceTB/SmolVLM-Instruct"
}

# 按分隔符 <|im_start|>assistant 和 <|im_end|> 提取最后一段内容
def split_lm2_content(text:str) -> str:
    tag_str = "<|im_start|>assistant"
    if tag_str in text:
        ret_str = text.split(tag_str)[-1].strip()
        return ret_str.replace("<|im_end|>", "")
    else:
        return text

# 按分隔符 `Assistant:` 提取最后一段内容
def split_vlm_content(text:str) -> str:
    tag_str = "Assistant:"
    if tag_str in text:
        ret_str = text.split(tag_str)[-1].strip()
        return ret_str
    else:
        return text

class LS_Load_SmolLM2_Model:
    def __init__(self):
        self.NODE_NAME = 'LoadSmolLM2Model'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        smollm2_model_list = list(smollm2_repo.keys())
        device_list = ['cuda', 'cpu']
        dtype_list = ["bf16", "fp32"]
        return {"required":
            {
                "model": (smollm2_model_list,),
                "dtype":(dtype_list,),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("SmolLM2_MODEL",)
    RETURN_NAMES = ("smolLM2_model", )
    FUNCTION = "load_smollm2_model"
    CATEGORY = '😺dzNodes/LayerUtility'

    def load_smollm2_model(self, model, dtype, device):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        repo_id = smollm2_repo[model]
        model_path = os.path.join("smol", model)
        model_path = check_and_download_model(model_path, repo_id)

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)

        return ({"tokenizer":tokenizer, "model":model, "dtype":dtype, "device":device},)

class LS_Load_SmolVLM_Model:
    def __init__(self):
        self.NODE_NAME = 'LoadSmolVLMModel'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        smolvlm_model_list = list(smolvlm_repo.keys())
        device_list = ['cuda', 'cpu']
        dtype_list = ["bf16", "fp32"]
        return {"required":
            {
                "model": (smolvlm_model_list,),
                "dtype":(dtype_list,),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("SmolVLM_MODEL",)
    RETURN_NAMES = ("smolVLM_model", )
    FUNCTION = "load_smolvlm_model"
    CATEGORY = '😺dzNodes/LayerUtility'

    def load_smolvlm_model(self, model, dtype, device):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

        repo_id = smolvlm_repo[model]
        model_path = os.path.join("smol", model)
        model_path = check_and_download_model(model_path, repo_id)

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        # Initialize processor and model
        processor = AutoProcessor.from_pretrained(model_path)

        try:
            import flash_attn
            use_flash_attention = device == "cuda" and torch_dtype == torch.bfloat16
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                _attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            ).to(device)
        except ImportError as e:
            print(e, ", use 'eager' instead.")
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                _attn_implementation="eager",
            ).to(device)

        return ({"processor":processor, "model":model, "dtype":dtype, "device":device},)


class LS_SmolLM2:

    def __init__(self):
        self.NODE_NAME = 'SmolLM2'
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {"required":{
                    "smolLM2_model": ("SmolLM2_MODEL",),
                    "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                    "do_sample": ("BOOLEAN", {"default": True}),
                    "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.1}),
                    "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                    "system_prompt": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": False}),
                    "user_prompt": ("STRING", {"default": "who are you?", "multiline": True}),
                },
                "optional": {
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text", )
    FUNCTION = "smollm2"
    CATEGORY = '😺dzNodes/LayerUtility'

    def smollm2(self, smolLM2_model, max_new_tokens, do_sample, temperature, top_p, system_prompt, user_prompt):

        tokenizer = smolLM2_model["tokenizer"]
        model = smolLM2_model["model"]
        device = smolLM2_model["device"]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
        ret_text = tokenizer.decode(outputs[0])
        ret_text = split_lm2_content(ret_text)
        log(f"{self.NODE_NAME} response is: {ret_text}")
        return (ret_text,)

class LS_SmolVLM:

    def __init__(self):
        self.NODE_NAME = 'SmolVLM'
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {"required":{
                    "image": ("IMAGE",),
                    "smolVLM_model": ("SmolVLM_MODEL",),
                    "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                    "user_prompt": ("STRING", {"default": "describe this image", "multiline": True}),
                },
                "optional": {
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text", )
    FUNCTION = "smolvlm"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = '😺dzNodes/LayerUtility'

    def smolvlm(self, image, smolVLM_model, max_new_tokens, user_prompt):

        processor = smolVLM_model["processor"]
        model = smolVLM_model["model"]
        device = smolVLM_model["device"]

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        ret_text = []
        for i in image:
            img = tensor2pil(i).convert("RGB")
            # Prepare inputs
            inputs = processor(text=prompt, images=[img], return_tensors="pt")
            inputs = inputs.to(device)
            # Generate outputs
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            result = generated_texts[0]
            result = split_vlm_content(result)
            ret_text.append(result)
            log(f"{self.NODE_NAME} response is: {ret_text}")

        return (ret_text,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: LoadSmolLM2Model": LS_Load_SmolLM2_Model,
    "LayerUtility: LoadSmolVLMModel": LS_Load_SmolVLM_Model,
    "LayerUtility: SmolLM2": LS_SmolLM2,
    "LayerUtility: SmolVLM": LS_SmolVLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: LoadSmolLM2Model": "LayerUtility: Load SmolLM2 Model(Advance)",
    "LayerUtility: LoadSmolVLMModel": "LayerUtility: Load SmolVLM Model(Advance)",
    "LayerUtility: SmolLM2": "LayerUtility: SmolLM2(Advance)",
    "LayerUtility: SmolVLM": "LayerUtility: SmolVLM(Advance)",
}

