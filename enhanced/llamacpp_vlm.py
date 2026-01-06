import os
import gc
import torch
import numpy as np
import logging
import threading
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler,
    Qwen25VLChatHandler, Qwen3VLChatHandler
)
import modules.config as config
from enhanced.logger import format_name
import ldm_patched.modules.model_management

logger = logging.getLogger(format_name(__name__))

class LlamaCppVLM:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.lock = threading.Lock()
        self.current_model_path = None
        self.current_chat_handler_name = None

    def get_chat_handler_class(self, name):
        handlers = {
            "Qwen3-VL": Qwen3VLChatHandler,
            "Qwen2.5-VL": Qwen25VLChatHandler,
            "LLaVA-1.5": Llava15ChatHandler,
            "LLaVA-1.6": Llava16ChatHandler,
            "Moondream2": MoondreamChatHandler,
            "nanoLLaVA": NanoLlavaChatHandler,
            "llama3-Vision-Alpha": Llama3VisionAlphaChatHandler,
            "MiniCPM-v2.6": MiniCPMv26ChatHandler,
            "MiniCPM-v4": MiniCPMv26ChatHandler,
        }
        return handlers.get(name)

    def _get_layer_count(self, path):
        import struct
        def read_u32(f):
            return struct.unpack("<I", f.read(4))[0]
        def read_u64(f):
            return struct.unpack("<Q", f.read(8))[0]
        def read_string(f):
            ln = read_u64(f)
            return f.read(ln).decode("utf-8")
        def read_value(f):
            vtype = read_u32(f)
            if vtype == 0: return struct.unpack("<B", f.read(1))[0]
            if vtype == 1: return struct.unpack("<b", f.read(1))[0]
            if vtype == 2: return struct.unpack("<H", f.read(2))[0]
            if vtype == 3: return struct.unpack("<h", f.read(2))[0]
            if vtype == 4: return struct.unpack("<I", f.read(4))[0]
            if vtype == 5: return struct.unpack("<i", f.read(4))[0]
            if vtype == 6: return struct.unpack("<f", f.read(4))[0]
            if vtype == 7: return struct.unpack("<?", f.read(1))[0]
            if vtype == 8: return read_string(f)
            if vtype == 9:
                atype = read_u32(f)
                count = read_u64(f)
                return [read_value_of_type(f, atype) for _ in range(count)]
            if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
            if vtype == 11: return struct.unpack("<q", f.read(8))[0]
            if vtype == 12: return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"Unknown value type {vtype}")
        def read_value_of_type(f, atype):
            if atype == 0: return struct.unpack("<B", f.read(1))[0]
            if atype == 1: return struct.unpack("<b", f.read(1))[0]
            if atype == 2: return struct.unpack("<H", f.read(2))[0]
            if atype == 3: return struct.unpack("<h", f.read(2))[0]
            if atype == 4: return struct.unpack("<I", f.read(4))[0]
            if atype == 5: return struct.unpack("<i", f.read(4))[0]
            if atype == 6: return struct.unpack("<f", f.read(4))[0]
            if atype == 7: return struct.unpack("<?", f.read(1))[0]
            if atype == 8: return read_string(f)
            if atype == 10: return struct.unpack("<Q", f.read(8))[0]
            if atype == 11: return struct.unpack("<q", f.read(8))[0]
            if atype == 12: return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"Unknown array item type {atype}")

        try:
            with open(path, "rb") as f:
                if f.read(4) != b"GGUF":
                    raise ValueError("Not a GGUF file")
                version = read_u32(f)
                tensor_count = read_u64(f)
                kv_count = read_u64(f)
                for _ in range(kv_count):
                    key = read_string(f)
                    value = read_value(f)
                    if key.lower().endswith(".block_count"):
                        return int(value)
        except Exception as e:
            logger.debug(f"Fast GGUF parse failed: {e}. Trying GGUFReader...")
            try:
                from gguf import GGUFReader
                reader = GGUFReader(path)
                for key in reader.fields.keys():
                    if key.endswith(".block_count") or key == "block_count":
                        return int(reader.get_field(key).parts[-1][0])
            except Exception as e2:
                logger.error(f"GGUFReader also failed: {e2}")
        return 32

    def load_model(self, model_name, chat_handler_name, n_gpu_layers=-1, n_ctx=8192):
        with self.lock:
            model_path = os.path.join(config.paths_LLM[0], model_name)
            if self.llm is not None and self.current_model_path == model_path and self.current_chat_handler_name == chat_handler_name:
                return

            self.free_model()

            logger.info(f"Loading Main LLM from: {model_path}")

            handler_class = self.get_chat_handler_class(chat_handler_name)
            mmproj_path = None
            if handler_class:
                model_dir = os.path.dirname(model_path)
                if os.path.exists(model_dir):
                    for f in os.listdir(model_dir):
                        if "mmproj" in f.lower() and f.endswith(".gguf"):
                            mmproj_path = os.path.join(model_dir, f)
                            break

                if mmproj_path:
                    logger.info(f"Using mmproj: {mmproj_path}")
                    try:
                        self.chat_handler = handler_class(clip_model_path=mmproj_path, verbose=False)
                    except TypeError:
                        self.chat_handler = handler_class(verbose=False)
                else:
                    logger.warning(f"No mmproj file found in {model_dir}. Some models may fail to load.")
                    self.chat_handler = handler_class(verbose=False)

            # Auto calculate n_gpu_layers if it's -1
            if n_gpu_layers == -1:
                try:
                    # Get free VRAM in GB
                    free_vram_bytes = ldm_patched.modules.model_management.get_free_memory()
                    vram_limit_gb = free_vram_bytes / (1024 ** 3)

                    # Buffer to prevent OOM (0.6GB)
                    vram_buffer = 0.6
                    available_vram_gb = vram_limit_gb - vram_buffer

                    if available_vram_gb > 0:
                        total_layers = self._get_layer_count(model_path)
                        # GGUF size estimation with 1.55 overhead factor
                        gguf_size_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
                        layer_size_gb = gguf_size_gb / total_layers

                        if mmproj_path:
                            mmproj_size_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                            n_gpu_layers = max(1, int((available_vram_gb - mmproj_size_gb) / layer_size_gb))
                        else:
                            n_gpu_layers = max(1, int(available_vram_gb / layer_size_gb))

                        n_gpu_layers = min(n_gpu_layers, total_layers)

                        # logger.info(f"Free: {vram_limit_gb:.2f}GB, Available: {available_vram_gb:.2f}GB")
                        # logger.info(f"Model: {os.path.basename(model_path)}, Layers: {total_layers}, LayerSize: {layer_size_gb*1024:.2f}MB")
                        # if mmproj_path:
                        #     logger.info(f"MMProj: {os.path.basename(mmproj_path)}, EstimatedSize: {mmproj_size_gb*1024:.2f}MB")
                        logger.info(f"Result: n_gpu_layers = {n_gpu_layers}")
                    else:
                        logger.warning(f"Not enough VRAM available ({vram_limit_gb:.2f}GB). Using CPU.")
                        n_gpu_layers = 0
                except Exception as e:
                    logger.warning(f"Calculation failed: {e}. Using default -1.")
                    n_gpu_layers = -1

            self.llm = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            self.current_model_path = model_path
            self.current_chat_handler_name = chat_handler_name
            ldm_patched.modules.model_management.print_memory_info("after load llama.cpp model")

    def free_model(self):
        if self.llm:
            self.llm.close()
            self.llm = None
        if self.chat_handler:
            try:
                self.chat_handler._exit_stack.close()
            except:
                pass
            self.chat_handler = None
        self.current_model_path = None
        self.current_chat_handler_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inference(self, image, prompt, chat_handler_override=None, max_tokens=1024, temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.1, seed=-1):
        if self.llm is None:
            logger.error("Model not loaded")
            return "Error: Model not loaded"

        import io
        import base64

        if chat_handler_override and self.current_chat_handler_name != chat_handler_override:
             logger.info(f"Inference with chat_handler_override: {chat_handler_override}")

        def image_to_base64(img_np):
            img = Image.fromarray(img_np)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        messages = []
        system_msg = "You are a helpful assistant. Follow instructions precisely. For any task (captioning, translation, expansion), output ONLY the result. Do not include any preamble, introduction, explanation, or conversational filler."
        messages.append({"role": "system", "content": system_msg})

        if image is not None:
            user_content = []
            user_content.append({"type": "text", "text": prompt})
            
            if isinstance(image, np.ndarray):
                base64_image = image_to_base64(image)
            elif isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                base64_image = None
            
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        logger.info(f"LlamaCpp Inference: prompt={prompt[:50]}... (image={'Yes' if image is not None else 'No'})")
        
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                seed=seed if seed != -1 else None
            )
            result = output['choices'][0]['message']['content']
            return result.strip()
        except Exception as e:
            logger.error(f"LlamaCpp Inference Error: {str(e)}")
            return f"Error during inference: {str(e)}"

llamacpp_vlm = LlamaCppVLM()
