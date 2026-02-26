import os
import gc
import torch
import numpy as np
import logging
import threading
from PIL import Image

from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

def setup_cuda_environment():
    """
    Setup CUDA environment variables to prioritize portable CUDA and avoid version mismatches.
    """
    import platform
    import glob

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    portable_root = os.path.dirname(project_root)

    cuda_paths = []
    is_portable = False

    if os.name == 'nt':
        python_embeded = os.path.join(portable_root, 'python_embeded')
        if os.path.exists(python_embeded):
            is_portable = True
            site_packages = os.path.join(python_embeded, 'Lib', 'site-packages')
            if os.path.exists(site_packages):
                nvidia_paths = glob.glob(os.path.join(site_packages, 'nvidia', '*', 'bin'))
                cuda_paths.extend(nvidia_paths)

                torch_lib = os.path.join(site_packages, 'torch', 'lib')
                if os.path.exists(torch_lib):
                    cuda_paths.append(torch_lib)

            bin_path = os.path.join(python_embeded, 'bin')
            if os.path.exists(bin_path):
                cuda_paths.append(bin_path)
    else:
        try:
            import sys
            for path in sys.path:
                if 'site-packages' in path:
                    nvidia_paths = glob.glob(os.path.join(path, 'nvidia', '*', 'lib'))
                    if nvidia_paths:
                        is_portable = True
                        cuda_paths.extend(nvidia_paths)
        except Exception as e:
            logger.debug(f"Failed to search site-packages for CUDA: {e}")

    if is_portable and cuda_paths:
        logger.info(f"Detected portable environment. Prioritizing CUDA paths: {cuda_paths}")

        for env_var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]:
            if env_var in os.environ:
                logger.info(f"Unsetting global {env_var}={os.environ[env_var]} to force portable CUDA usage")
                del os.environ[env_var]

        if os.name == 'nt':
            current_path = os.environ.get("PATH", "")
            new_path = ";".join(cuda_paths) + ";" + current_path
            os.environ["PATH"] = new_path
        else:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            new_ld = ":".join(cuda_paths) + (":" + current_ld if current_ld else "")
            os.environ["LD_LIBRARY_PATH"] = new_ld

    if not is_portable:
        if os.name == 'nt':
            if "CUDA_PATH" in os.environ:
                _cuda_path = os.environ["CUDA_PATH"]
                _cuda_bin = os.path.join(_cuda_path, "bin")
                if not os.path.exists(_cuda_bin):
                    del os.environ["CUDA_PATH"]
                    logger.info("Removed invalid CUDA_PATH from environment")
        else:
            std_cuda_paths = ["/usr/local/cuda/lib64"]
            try:
                found_paths = glob.glob("/usr/local/cuda-*/lib64")
                if found_paths:
                    std_cuda_paths.extend(sorted(found_paths, reverse=True))
            except:
                pass

            for path in std_cuda_paths:
                if os.path.exists(path):
                    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                    if path not in current_ld:
                        os.environ["LD_LIBRARY_PATH"] = path + (":" + current_ld if current_ld else "")
                        logger.info(f"Added system CUDA path {path} to LD_LIBRARY_PATH")

    if not os.name == 'nt':
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "cuda-13" in ld_path and "cuda-12" not in ld_path:
            has_12 = False
            for p in ld_path.split(":"):
                if p and os.path.exists(os.path.join(p, "libcublas.so.12")):
                    has_12 = True
                    break
            if not has_12:
                logger.warning("Detected CUDA 13 but libcublas.so.12 is missing. Clearing CUDA paths to avoid crash.")
                new_ld = ":".join([p for p in ld_path.split(":") if "cuda" not in p.lower()])
                os.environ["LD_LIBRARY_PATH"] = new_ld

setup_cuda_environment()

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
        NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler,
        Qwen25VLChatHandler, Qwen3VLChatHandler
    )
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import llama_cpp: {e}")
    logger.error("Please ensure CUDA libraries are correctly installed and in your library path.")
    Llama = None
    Llava15ChatHandler = Llava16ChatHandler = MoondreamChatHandler = None
    NanoLlavaChatHandler = Llama3VisionAlphaChatHandler = MiniCPMv26ChatHandler = None
    Qwen25VLChatHandler = Qwen3VLChatHandler = None
    LLAMA_CPP_AVAILABLE = False

import modules.config as config
import ldm_patched.modules.model_management

logger = logging.getLogger(format_name(__name__))

class LlamaCppVLM:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.lock = threading.RLock()
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
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python is not correctly installed or CUDA libraries are missing.")
            return

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
        with self.lock:
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
        with self.lock:
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
                
                images = image if isinstance(image, (list, tuple)) else [image]
                for img in images:
                    if img is None:
                        continue
                    if isinstance(img, np.ndarray):
                        base64_image = image_to_base64(img)
                    elif isinstance(img, Image.Image):
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG", quality=85)
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
