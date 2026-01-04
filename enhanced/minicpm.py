import os
import gc
import torch
import shared
import threading
import numpy as np
import modules.config as config
import modules.flags as flags
import enhanced.translator as translator
import enhanced.superprompter as superprompter
import ldm_patched.modules.model_management
import modules.default_pipeline as pipeline
import enhanced.all_parameters as ads
import logging
from enhanced.llamacpp_vlm import llamacpp_vlm
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

from PIL import Image
from transformers import AutoTokenizer, AutoModel
from modules.model_loader import download_diffusers_model
from modules.util import HWC3, resize_image, is_chinese
from enhanced.simpleai import comfyd, p2p_task

class MiniCPM:
    prompt_i2t = "A descriptive caption for this image. Output only the caption text without any preamble or explanation."
    output_chinese = "and output it in Chinese. Only provide the Chinese text, no other explanation."
    prompt_extend = "Expand the following description to obtain a descriptive caption with more details in image. Output only the expanded description without any preamble or explanation: "
    prompt_translator = "Translate the following text into English. Output only the translation itself, no other text or explanation:"
    prompt_translator_cn = "Translate the following text into Chinese. Output only the translation itself, no other text or explanation:"

    # 版本配置定义
    VERSIONS = {
        "MiniCPMv26": {
            "model": "MiniCPMv2_6-prompt-generator",
            "model_file": "pytorch_model-00001-of-00002.bin",
            "model_url": "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/models_minicpm_v2.6_prompt_simpleai_1224.zip",
            "is_llamacpp": False
        },
        "MiniCPMv45": {
            "model": "MiniCPM-V-4_5-int4",
            "model_file": "model-00001-of-00002.safetensors",
            "model_url": "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/MiniCPM-V-4_5-int4.zip",
            "is_llamacpp": False
        },
        "Qwen3-VL-4B-Instruct-abliterated": {
            "model": "Qwen3-VL-4B-Instruct-abliterated",
            "model_file": "Qwen3-VL-4B-Instruct-abliterated",
            "model_urls": {
                "Qwen3-VL-4B-Instruct-abliterated-v1.Q8_0.gguf": "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-4B-Instruct-abliterated/Qwen3-VL-4B-Instruct-abliterated-v1.Q8_0.gguf",
                "Qwen3-VL-4B-Instruct-abliterated-v1.mmproj-Q8_0.gguf": "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-4B-Instruct-abliterated/Qwen3-VL-4B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
            },
            "is_llamacpp": True
        },
        "Qwen3-VL-8B-Instruct-abliterated": {
            "model": "Qwen3-VL-8B-Instruct-abliterated",
            "model_file": "Qwen3-VL-8B-Instruct-abliterated",
            "model_urls": {
                "Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf": "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-8B-Instruct-abliterated/Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
                "Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf": "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-8B-Instruct-abliterated/Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf"
            },
            "is_llamacpp": True
        }
    }

    # 运行时参数（由 set_version 统一管理）
    model = ""
    model_file = ""
    model_url = None
    model_urls = {}
    is_llamacpp = False
    current_version = ""

    remove_prefixs = [
        'A descriptive caption for this image could be: "',
        '"',
        ]

    lock = threading.Lock()
    model_cpm = None
    tokenizer = None
    enable = ads.get_admin_default('minicpm_checkbox')
    bf16_support = ( torch.cuda.is_available() and torch.cuda.get_device_capability(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))[0] >= 8 )

    # 状态标记
    is_processing = False
    processing_lock = threading.Lock()

    @classmethod
    def set_version(cls, version):
        if not version or version == 'None':
            version = "MiniCPMv26"

        config_data = cls.VERSIONS.get(version)
        if not config_data:
            logger.warning(f"未知 VLM 版本: {version}。回退到 MiniCPMv26")
            version = "MiniCPMv26"
            config_data = cls.VERSIONS[version]

        with cls.lock:
            cls.current_version = version
            cls.model = config_data["model"]
            cls.is_llamacpp = config_data.get("is_llamacpp", False)
            cls.model_url = config_data.get("model_url")
            cls.model_urls = config_data.get("model_urls", {})
            cls.model_file = os.path.join(cls.model, config_data["model_file"])

            logger.info(f"设置 VLM 模型: 版本={version}, 模型路径={cls.model}, is_llamacpp={cls.is_llamacpp}")

    def __init__(self):
        pass

    @classmethod
    def set_enable(cls, flag):
        with cls.lock:
            cls.enable = flag

    @classmethod
    def get_enable(cls):
        return cls.enable

    @classmethod
    def set_processing_status(cls, status):
        with cls.processing_lock:
            previous_status = cls.is_processing
            cls.is_processing = status
            if previous_status != status:
                logger.debug(f"VLM processing status changed to: {'processing' if status else 'idle'}")

    @classmethod
    def get_processing_status(cls):
        with cls.processing_lock:
            return cls.is_processing
    def load_model(self, download=False):
        if MiniCPM.is_llamacpp:
            # Check for multi-file download (Qwen3-VL style)
            if MiniCPM.model_urls:
                model_dir = os.path.join(config.paths_LLM[0], MiniCPM.model)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)

                all_files_exist = True
                for file_name in MiniCPM.model_urls:
                    file_path = os.path.join(model_dir, file_name)
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        break

                if not all_files_exist:
                    if download:
                        from modules.model_loader import load_file_from_url
                        logger.info(f"正在为 {MiniCPM.current_version} 下载模型文件...")
                        for file_name, url in MiniCPM.model_urls.items():
                            load_file_from_url(url, model_dir=model_dir, file_name=file_name)
                    else:
                        logger.warning(f"模型文件缺失，自动下载失败: {model_dir}")
                        return

            # For llama.cpp, we use the llamacpp_vlm module
            model_dir = os.path.join(config.paths_LLM[0], MiniCPM.model)
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return

            gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf') and "mmproj" not in f.lower()]
            if not gguf_files:
                logger.error(f"No .gguf file found in {model_dir}")
                return

            model_file = os.path.join(MiniCPM.model, gguf_files[0])
            chat_handler_name = "Qwen3-VL"
            llamacpp_vlm.load_model(model_file, chat_handler_name)
            return

        if not shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file):
            if download:
                download_diffusers_model('llms', MiniCPM.model, 2, MiniCPM.model_url)
            else:
                return
        import sys
        from typing import List
        sys.modules[__name__].__builtins__['List'] = List
        MODEL_PATH = os.path.join(config.paths_llms[0], MiniCPM.model)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=False, local_files_only=True, device_map="cpu")
        text_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=False, local_files_only=True,
                attn_implementation="sdpa", dtype=torch.bfloat16 if MiniCPM.bf16_support else torch.float16, device_map="cpu")
        text_model.eval()
        with MiniCPM.lock:
            MiniCPM.model_cpm = text_model
            MiniCPM.tokenizer = tokenizer
        ldm_patched.modules.model_management.print_memory_info("after load minicpm model")
        return

    def free_model(self):
        llamacpp_vlm.free_model()
        if MiniCPM.model_cpm is None and MiniCPM.tokenizer is None:
            return
        with MiniCPM.lock:
            del MiniCPM.model_cpm
            del MiniCPM.tokenizer
            MiniCPM.model_cpm = None
            MiniCPM.tokenizer = None
        translator.free_translator_model()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        ldm_patched.modules.model_management.print_memory_info("after free minicpm model")

    def inference(self, image, prompt, max_tokens=2048, temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.05, seed=-1):
        # 设置为处理中状态
        MiniCPM.set_processing_status(True)
        logger.debug("Starting VLM local inference...")
        try:
            if ads.get_admin_default('p2p_active_checkbox') and ads.get_admin_default('p2p_remote_process').lower()=='out':
                if isinstance(image, np.ndarray):
                    image = p2p_task.ndarray_to_webp_bytes(image)
                args = (image, prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, seed)
                task = p2p_task.AsyncTask(method="minicpm_inference", args=args)
                p2p_task.request_p2p_task(task)
                result = task.wait(30)
                return result[0]
            else:
                return self.inference_local(image, prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, seed)
        finally:
            # 无论成功还是失败，都设置为非处理中状态
            MiniCPM.set_processing_status(False)
            logger.debug("VLM local inference completed")

    @torch.no_grad()
    @torch.inference_mode()
    def inference_local(self, image, prompt, max_tokens=2048, temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.05, seed=-1):
        try:
            # 设置处理状态为True
            self.set_processing_status(True)
            logger.debug("VLM inference_local started")

            comfyd.stop()
            pipeline.free_everything()
            ldm_patched.modules.model_management.print_vram_info_by_nvml("before minicpm inference")

            if MiniCPM.is_llamacpp:
                chat_handler_name = "Qwen3-VL"
                if llamacpp_vlm.llm is None:
                    self.load_model(download=True)
                res = llamacpp_vlm.inference(
                    image=image,
                    prompt=prompt,
                    chat_handler_override=chat_handler_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    seed=seed
                )
                return res

            if MiniCPM.model_cpm is None or MiniCPM.tokenizer is None:
                self.load_model(download=True)

            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                device = torch.device('cuda')
                MiniCPM.model_cpm = MiniCPM.model_cpm.to(device)
            else:
                device = torch.device('cpu')

            image = image if image is None else Image.fromarray(resize_image(image, min_side=768, resize_mode=3))
            msgs = [{'role': 'user', 'content': [image, prompt]}]

            res = MiniCPM.model_cpm.chat(
                image=None,
                msgs=msgs,
                tokenizer=MiniCPM.tokenizer,
                sampling=True,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )

            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                MiniCPM.model_cpm = MiniCPM.model_cpm.to('cpu')

            generated_text = res
            logger.debug(f'The generated text:{generated_text}')
            ldm_patched.modules.model_management.print_memory_info("after minicpm inference")
            return generated_text
        finally:
            self.set_processing_status(False)
            logger.debug("VLM inference_local finished")

    def interrogate(self, image, output_chinese=False, prompt=None, additional_prompt=None):
        MiniCPM.set_processing_status(True)
        try:
            if prompt is not None:
                logger.debug(f'The prompt of image: {prompt}')
                return self.inference(image, prompt)
            if additional_prompt:
                prompt = additional_prompt
            else:
                prompt = MiniCPM.prompt_i2t

            if output_chinese:
                prompt = f'{prompt}, {MiniCPM.output_chinese}'
            logger.debug(f'The prompt of image: {prompt}')
            result_prompt = self.inference(image, prompt)

            for prefix in MiniCPM.remove_prefixs:
                if result_prompt.startswith(prefix):
                    result_prompt = result_prompt[len(prefix):]
            if result_prompt.endswith('"'):
                result_prompt = result_prompt[:-1]
            return result_prompt
        finally:
            MiniCPM.set_processing_status(False)

    def model_exists(self):
        if MiniCPM.is_llamacpp:
            model_dir = os.path.join(config.paths_LLM[0], MiniCPM.model)
            return os.path.exists(model_dir)
        return shared.modelsinfo.exists_model(catalog="llms", model_path=MiniCPM.model_file)

    def extended_prompt(self, input_text, prompt, input_image, state, translation_methods='Third APIs'):
        if 'scene_frontend' in state:
            scenes = state['scene_frontend']
            theme = state['scene_theme']
            prompt_prompt = flags.get_value_by_scene_theme(state, theme, 'agent_prompt', '')
            if prompt_prompt:
                if MiniCPM.get_enable():
                    logger.debug(f"Using {'LlamaCpp' if MiniCPM.is_llamacpp else 'MiniCPM'} for scene extended prompt")
                    return self.interrogate(input_image, prompt=f'{prompt_prompt}{input_text}')
                else:
                    return input_text
        else:
            if not MiniCPM.get_enable() or not self.model_exists():
                return superprompter.answer(input_text=translator.convert(f'{prompt}{input_text}', translation_methods))
            else:
                logger.debug(f"Using {'LlamaCpp' if MiniCPM.is_llamacpp else 'MiniCPM'} for standard extended prompt")
                return self.inference(None, prompt=f'{MiniCPM.prompt_extend}{input_text}')

    def translate(self, input_text, method=None):
        if not is_chinese(input_text):
            return input_text
        if MiniCPM.get_enable() and self.model_exists() and method in [None, 'Big Model']:
            logger.debug(f"Using {'LlamaCpp' if MiniCPM.is_llamacpp else 'MiniCPM'} for translation to English")
            return self.inference(None, prompt=f'{MiniCPM.prompt_translator}{input_text}')
        else:
            return translator.convert(input_text, method)

    def translate_cn(self, input_text, method=None):
        if is_chinese(input_text):
            return input_text
        if MiniCPM.get_enable() and self.model_exists() and method in [None, 'Big Model']:
            logger.debug(f"Using {'LlamaCpp' if MiniCPM.is_llamacpp else 'MiniCPM'} for translation to Chinese")
            return self.inference(None, prompt=f'{MiniCPM.prompt_translator_cn}{input_text}')
        else:
            return translator.convert(input_text, method)
       
# 初始化模型版本
MiniCPM.set_version(ads.get_admin_default('minicpm_version'))

minicpm = MiniCPM()
default_interrogator = minicpm.interrogate

