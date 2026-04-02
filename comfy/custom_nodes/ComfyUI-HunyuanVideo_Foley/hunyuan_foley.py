import os
import torch
import random
import numpy as np
import folder_paths
import torchaudio
import gc
import logging
import sys
import contextlib
import warnings
from PIL import Image

# --- Suppress noisy logs and warnings ---
warnings.filterwarnings("ignore")

# Force diffusers and transformers to be quiet as early as possible
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

try:
    import diffusers.utils.logging
    diffusers.utils.logging.set_verbosity_error()
except ImportError:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

class DummyFile:
    def write(self, x): pass
    def flush(self): pass
    def isatty(self): return False

@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# --- Workaround for diffusers bug ---
# Fix for: NameError: name 'logger' is not defined in diffusers.quantizers.torchao.torchao_quantizer
# We do this WITHOUT suppression first to avoid capturing a closed file in the logger
try:
    import diffusers.utils.logging
    import builtins
    if not hasattr(builtins, 'logger'):
        # Inject a logger that is already set to ERROR
        l = diffusers.utils.logging.get_logger("diffusers.quantizers.torchao.torchao_quantizer")
        l.setLevel(logging.ERROR)
        builtins.logger = l
except Exception:
    pass

from comfy_api.latest import ComfyExtension, io, ui

# Use relative imports for our vendored code
from .src.hunyuanvideo_foley.utils.model_utils import denoise_process
from .src.hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
from .src.hunyuanvideo_foley.utils.feature_utils import encode_video_with_sync, encode_text_feat
from .src.hunyuanvideo_foley.constants import FPS_VISUAL
from .src.hunyuanvideo_foley.models.dac_vae.model.dac import DAC
from .download_models import ensure_models_downloaded, ensure_vae_downloaded

# --- Helper Functions ---
logging.basicConfig(level=logging.INFO, format='HunyuanFoley (%(levelname)s): %(message)s')

def set_manual_seed(seed):
    seed = int(seed)
    numpy_seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_state_dict(model, model_path):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model

@contextlib.contextmanager
def model_on_device(model, device, dtype=None):
    """Context manager: temporarily move a model to `device`, then return it to CPU."""
    if dtype is not None:
        model.to(device, dtype=dtype)
    else:
        model.to(device)
    try:
        yield model
    finally:
        model.to("cpu")
        empty_cuda_cache()

def install_block_offload_hooks(foley_model, device, dtype):
    """
    Optimized block-level CPU↔GPU offloading for foley_model:

    Strategy:
      1. Small auxiliary modules (embedders, projections, final layer …) and
         root-level parameters always reside on GPU during inference.
      2. Transformer blocks (triple_blocks + single_blocks) are greedily
         preloaded into VRAM as many as fit, leaving ~1.5 GB headroom for
         activations.  Blocks that don't fit are kept on CPU.
      3. For the CPU-resident blocks, forward pre/post hooks swap them to GPU
         one at a time.  A dedicated CUDA prefetch stream starts transferring
         block[i+1] while block[i] is still computing, overlapping data
         transfer with GPU compute and maximising GPU utilisation.

    All tensors remain in real CPU RAM (no meta tensors), so helper methods
    like get_empty_clip_sequence() keep working correctly.
    """
    block_list_names = {'triple_blocks', 'single_blocks'}

    # --- Step 1: small child modules → GPU ---
    small_modules = {}
    for name, child in foley_model.named_children():
        if name not in block_list_names:
            small_modules[name] = child
    for name, mod in small_modules.items():
        mod.to(device, dtype=dtype)

    # --- Step 2: root-level nn.Parameters / buffers → GPU ---
    # (e.g. sync_pos_emb, empty_clip_feat, empty_sync_feat)
    root_param_names = list(foley_model._parameters.keys())
    root_buffer_names = list(foley_model._buffers.keys())
    for pname in root_param_names:
        p = foley_model._parameters[pname]
        if p is not None:
            foley_model._parameters[pname] = torch.nn.Parameter(
                p.to(device, dtype=dtype), requires_grad=p.requires_grad
            )
    for bname in root_buffer_names:
        b = foley_model._buffers[bname]
        if b is not None:
            foley_model._buffers[bname] = b.to(device, dtype=dtype)

    # --- Step 3: greedy block preloading ---
    all_blocks = list(foley_model.triple_blocks) + list(foley_model.single_blocks)

    preloaded_ids = set()
    offloaded_blocks = []

    if torch.cuda.is_available():
        # Query free VRAM now (after small modules are loaded)
        free_bytes = torch.cuda.mem_get_info(device)[0]
        # Reserve 1.5 GB for activations and intermediate tensors
        vram_budget = max(0, int(free_bytes - 1.5 * 1024 ** 3))

        cumulative = 0
        for block in all_blocks:
            block_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
            block_bytes += sum(b.numel() * b.element_size() for b in block.buffers())
            if cumulative + block_bytes <= vram_budget:
                block.to(device, dtype=dtype)
                preloaded_ids.add(id(block))
                cumulative += block_bytes
            else:
                offloaded_blocks.append(block)
    else:
        # Non-CUDA: all blocks must be offloaded (though there won't be GPU at all)
        offloaded_blocks = all_blocks

    # Record which block indices are preloaded (for restore on reuse)
    preloaded_indices = [i for i, blk in enumerate(all_blocks) if id(blk) in preloaded_ids]

    n_pre = len(preloaded_ids)
    n_off = len(offloaded_blocks)
    logging.info(
        f"CPU offload: {n_pre} blocks preloaded to GPU, "
        f"{n_off} blocks will be swapped with async prefetch."
    )

    # --- Step 4: hooks + async prefetch for offloaded blocks ---
    handles = []

    if offloaded_blocks and torch.cuda.is_available():
        # A dedicated stream for async prefetch transfers
        pfx_stream = torch.cuda.Stream(device=device)

        # Map block id → next offloaded block (for prefetch scheduling)
        next_offloaded = {}
        for i, blk in enumerate(offloaded_blocks):
            next_offloaded[id(blk)] = offloaded_blocks[i + 1] if i + 1 < len(offloaded_blocks) else None

        def make_pre_sync(ps):
            """Wait for the prefetch of *this* block to complete."""
            def pre_hook(module, args):
                # Ensure prefetch copy is visible to the compute stream
                torch.cuda.current_stream(device).wait_stream(ps)
                # Fallback: if somehow still on CPU (first block edge case),
                # load synchronously
                p = next(iter(module.parameters()), None)
                if p is not None and p.device.type == 'cpu':
                    module.to(device, dtype=dtype)
            return pre_hook

        def make_post_prefetch(blk_id, ps, nxt_map, dev, dt):
            """Move current block back to CPU, then prefetch the next one."""
            def post_hook(module, args, output):
                module.to('cpu')
                nxt = nxt_map.get(blk_id)
                if nxt is not None:
                    with torch.cuda.stream(ps):
                        nxt.to(dev, dtype=dt)
            return post_hook

        for blk in offloaded_blocks:
            h1 = blk.register_forward_pre_hook(make_pre_sync(pfx_stream))
            h2 = blk.register_forward_hook(
                make_post_prefetch(id(blk), pfx_stream, next_offloaded, device, dtype)
            )
            handles.extend([h1, h2])

        # Kick off the prefetch of the very first offloaded block right away
        with torch.cuda.stream(pfx_stream):
            offloaded_blocks[0].to(device, dtype=dtype)

    elif offloaded_blocks:
        # Non-CUDA fallback: simple synchronous swap
        def make_pre(dev, dt):
            def pre_hook(module, args):
                module.to(dev, dtype=dt)
            return pre_hook

        def make_post():
            def post_hook(module, args, output):
                module.to('cpu')
            return post_hook

        for blk in offloaded_blocks:
            h1 = blk.register_forward_pre_hook(make_pre(device, dtype))
            h2 = blk.register_forward_hook(make_post())
            handles.extend([h1, h2])

    return handles, small_modules, root_param_names, root_buffer_names, preloaded_indices, all_blocks


@contextlib.contextmanager
def foley_block_offload_context(foley_model, device, dtype):
    """
    Context manager that sets up optimised block offloading for one
    denoising pass and cleans up (returns everything to CPU) afterward.
    """
    try:
        if not hasattr(foley_model, '_block_offload_handles'):
            logging.info("CPU offload: setting up greedy block offloading + async prefetch...")
            foley_model.to(dtype=dtype)  # ensure correct dtype on CPU first
            handles, small_modules, root_params, root_buffers, pre_idx, all_blks = install_block_offload_hooks(
                foley_model, device, dtype
            )
            foley_model._block_offload_handles = handles
            foley_model._block_offload_small = list(small_modules.keys())
            foley_model._block_offload_root_params = root_params
            foley_model._block_offload_root_buffers = root_buffers
            foley_model._block_offload_preloaded_idx = pre_idx
            # Store a stable reference list of all blocks in forward order
            foley_model._block_offload_all_blocks = all_blks
        else:
            logging.info("CPU offload: reusing block offload setup (restoring GPU state)...")
            # Restore preloaded blocks to GPU
            for idx in getattr(foley_model, '_block_offload_preloaded_idx', []):
                all_blks = getattr(foley_model, '_block_offload_all_blocks', [])
                if idx < len(all_blks):
                    try:
                        all_blks[idx].to(device, dtype=dtype)
                    except Exception:
                        pass
            # Restore small modules to GPU
            for name in foley_model._block_offload_small:
                child = getattr(foley_model, name, None)
                if child is not None and hasattr(child, 'to'):
                    child.to(device, dtype=dtype)
            # Restore root parameters and buffers to GPU
            for pname in foley_model._block_offload_root_params:
                p = foley_model._parameters.get(pname)
                if p is not None:
                    foley_model._parameters[pname] = torch.nn.Parameter(
                        p.to(device, dtype=dtype), requires_grad=p.requires_grad
                    )
            for bname in foley_model._block_offload_root_buffers:
                b = foley_model._buffers.get(bname)
                if b is not None:
                    foley_model._buffers[bname] = b.to(device, dtype=dtype)
        yield foley_model
    finally:
        # Return ALL transformer blocks to CPU (covers both preloaded and hooked)
        for block in list(foley_model.triple_blocks) + list(foley_model.single_blocks):
            p = next(iter(block.parameters()), None)
            if p is not None and p.device.type != 'cpu':
                try:
                    block.to('cpu')
                except Exception:
                    pass
        # Return small child modules to CPU
        for name in getattr(foley_model, '_block_offload_small', []):
            child = getattr(foley_model, name, None)
            if child is not None and hasattr(child, 'to'):
                try:
                    child.to('cpu')
                except Exception:
                    pass
        # Return root-level parameters to CPU
        for pname in getattr(foley_model, '_block_offload_root_params', []):
            p = foley_model._parameters.get(pname)
            if p is not None:
                try:
                    foley_model._parameters[pname] = torch.nn.Parameter(
                        p.to('cpu'), requires_grad=p.requires_grad
                    )
                except Exception:
                    pass
        # Return root-level buffers to CPU
        for bname in getattr(foley_model, '_block_offload_root_buffers', []):
            b = foley_model._buffers.get(bname)
            if b is not None:
                try:
                    foley_model._buffers[bname] = b.to('cpu')
                except Exception:
                    pass
        empty_cuda_cache()

# --- Model Cache ---
loaded_models_cache = {}
loaded_vaes_cpu = {}

def unload_foley_models():
    """
    Fully unload all cached Hunyuan-Foley models from memory.

    In cpu_offload mode the models live in system RAM (or virtual memory).
    This function:
      1. Removes any block-offload forward hooks so closures don't hold
         references to blocks.
      2. Moves every sub-model to CPU (no-op if already there).
      3. Clears the global model cache so Python's GC can reclaim the RAM.
      4. Calls gc.collect() to trigger immediate collection.
    """
    global loaded_models_cache
    if not loaded_models_cache:
        return

    logging.info("Fully unloading Hunyuan-Foley models from memory...")
    try:
        for key in list(loaded_models_cache.keys()):
            model_tuple = loaded_models_cache[key]
            model_dict = model_tuple[0]

            foley_model = model_dict.get('foley_model')
            if foley_model is not None:
                # Remove block-offload hooks so their closures release block refs
                handles = getattr(foley_model, '_block_offload_handles', [])
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                for attr in ('_block_offload_handles', '_block_offload_small',
                             '_block_offload_root_params', '_block_offload_root_buffers',
                             '_block_offload_preloaded_idx', '_block_offload_all_blocks'):
                    try:
                        delattr(foley_model, attr)
                    except AttributeError:
                        pass

            # Move every sub-model to CPU so GPU memory is freed
            for model_name, model in list(model_dict.items()):
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except Exception:
                        pass

            # Nullify each sub-model to release object references so GC can
            # reclaim the RAM. Preserve __reload_params__ so the sampler can
            # auto-reload on the next run without user intervention.
            _keep = {'__reload_params__'}
            for k in list(model_dict.keys()):
                if k not in _keep:
                    model_dict[k] = None
            # Sentinel: sampler detects this and auto-reloads
            model_dict['__unloaded__'] = True

        loaded_models_cache.clear()
        empty_cuda_cache()
        gc.collect()
        logging.info("Hunyuan-Foley models fully unloaded from memory.")
    except Exception as e:
        logging.error(f"An error occurred while unloading models: {e}")


def unload_vae_models():
    """Fully unload cached DAC VAE models from memory."""
    global loaded_vaes_cpu
    if not loaded_vaes_cpu:
        return
    logging.info("Unloading Hunyuan-Foley VAE from memory...")
    for vae in loaded_vaes_cpu.values():
        try:
            del vae
        except Exception:
            pass
    loaded_vaes_cpu.clear()
    gc.collect()


class HunyuanFoleyModelLoader(io.ComfyNode):
    def __init__(self):
        self.model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        root_folder_name = os.path.basename(model_dir)
        checkpoints = ["hunyuanvideo_foley.pth", "hunyuanvideo_foley_xl.pth"]

        def has_checkpoint(path):
            return any(os.path.isfile(os.path.join(path, cp)) for cp in checkpoints)

        model_paths = []
        # Always include the root hunyuan_foley folder so users can select it
        # and trigger an automatic download even before any files exist.
        model_paths.append(root_folder_name)

        for f in os.listdir(model_dir):
            sub_path = os.path.join(model_dir, f)
            if os.path.isdir(sub_path) and has_checkpoint(sub_path):
                model_paths.append(f)

        return io.Schema(
            node_id="HunyuanFoleyModelLoader",
            display_name="Hunyuan-Foley model loader",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Combo.Input("model_path_name", options=model_paths),
                io.Combo.Input("foley_checkpoint_name", options=["hunyuanvideo_foley.pth", "hunyuanvideo_foley_xl.pth"]),
                io.Boolean.Input("cpu_offload", default=False,
                                 tooltip="将所有模型保留在系统内存(RAM)中，仅在推理时临时移至GPU。可大幅降低显存占用，但每步推理速度会略慢。")
            ],
            outputs=[
                io.Custom("FOLEY_MODEL").Output()
            ]
        )

    @classmethod
    def execute(cls, model_path_name, foley_checkpoint_name, cpu_offload=False, error=None) -> io.NodeOutput:
        global loaded_models_cache

        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        root_folder_name = os.path.basename(model_dir)
        if model_path_name == root_folder_name:
            model_path = model_dir
        else:
            model_path = os.path.join(model_dir, model_path_name)

        # Auto-download any missing model files before attempting to load
        ensure_models_downloaded(model_path, foley_checkpoint_name)

        precision = "bfloat16"
        cache_key = (os.path.normpath(model_path), precision, foley_checkpoint_name, bool(cpu_offload))

        models_to_unload = []
        for key, cached_tuple in loaded_models_cache.items():
            if key != cache_key:
                model_dict = cached_tuple[0]
                main_model = model_dict.get('foley_model')
                # cpu_offload hook models report device as 'cpu' on their params;
                # check the hook attribute instead
                is_on_cuda = (
                    hasattr(main_model, '_hf_hook') or
                    (main_model is not None and next(main_model.parameters(), None) is not None
                     and next(main_model.parameters()).device.type == 'cuda')
                )
                if main_model is not None and is_on_cuda:
                    models_to_unload.append(key)
        
        if models_to_unload:
            logging.info(f"Switching models. Unloading {len(models_to_unload)} model(s) from VRAM to CPU...")
            for key in models_to_unload:
                model_dict = loaded_models_cache[key][0]
                for model in model_dict.values():
                    if hasattr(model, 'to'):
                        try:
                            model.to('cpu')
                        except Exception:
                            pass
            empty_cuda_cache()

        if foley_checkpoint_name == "hunyuanvideo_foley_xl.pth":
            config_name = "hunyuanvideo-foley-xl.yaml"
        else:
            config_name = "hunyuanvideo-foley-xxl.yaml"
        
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "src/hunyuanvideo_foley/configs", config_name)
        
        if cache_key in loaded_models_cache:
            logging.info(f"Using cached model (cpu_offload={'enabled' if cpu_offload else 'disabled'}).")
            return io.NodeOutput(loaded_models_cache[cache_key])

        # When cpu_offload is enabled, load everything to CPU;
        # the sampler will move individual models to GPU on demand.
        if cpu_offload:
            target_device = "cpu"
            logging.info(f"Loading {foley_checkpoint_name} to CPU (cpu_offload mode)...")
        else:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading {foley_checkpoint_name} to {target_device}...")

        model_dict, cfg = cls.load_all_models_to_vram(model_path, config_path, precision, target_device, foley_checkpoint_name)
        
        # Store reload params so the sampler can auto-reload if models were
        # unloaded from memory (by DACHunyuanVAEDecode's unload_models_after_use)
        model_dict['__reload_params__'] = {
            'model_path_name': model_path_name,
            'foley_checkpoint_name': foley_checkpoint_name,
            'cpu_offload': bool(cpu_offload),
        }

        foley_model_tuple = (model_dict, cfg, precision, bool(cpu_offload))
        loaded_models_cache[cache_key] = foley_model_tuple
        return io.NodeOutput(foley_model_tuple)

    @classmethod
    def load_all_models_to_vram(cls, model_path, config_path, precision, target_device, foley_checkpoint_name):
        from .src.hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
        from .src.hunyuanvideo_foley.models.synchformer import Synchformer
        from transformers import AutoTokenizer, ClapTextModelWithProjection, SiglipImageProcessor, SiglipVisionModel
        
        cfg = load_yaml(config_path)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        siglip_path = os.path.join(model_path, "siglip2")
        clap_path = os.path.join(model_path, "clap")
        
        if not os.path.isdir(siglip_path): raise FileNotFoundError(f"SigLIP2 folder not found at {siglip_path}")
        if not os.path.isdir(clap_path): raise FileNotFoundError(f"CLAP folder not found at {clap_path}")

        foley_model = HunyuanVideoFoley(cfg, dtype=dtype).eval()
        with suppress_output():
            load_state_dict(foley_model, os.path.join(model_path, foley_checkpoint_name))
        
        with suppress_output():
            siglip2_model = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True, low_cpu_mem_usage=True).eval()
            siglip2_preprocess = SiglipImageProcessor.from_pretrained(siglip_path, local_files_only=True)

            clap_tokenizer = AutoTokenizer.from_pretrained(clap_path, local_files_only=True)
            clap_model = ClapTextModelWithProjection.from_pretrained(clap_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        
        from torchvision.transforms import v2
        
        # Preprocessor for PIL/Numpy inputs (legacy/backup)
        syncformer_preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), 
            v2.CenterCrop(224), 
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Preprocessor for Tensor inputs (B, C, H, W) in [0, 1]
        syncformer_preprocess_tensor = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        syncformer_model = Synchformer().eval()
        with suppress_output():
            syncformer_model.load_state_dict(torch.load(os.path.join(model_path, "synchformer_state_dict.pth"), map_location="cpu"))
        
        model_dict = { 
            'foley_model': foley_model.to(target_device, dtype=dtype), 
            'siglip2_preprocess': siglip2_preprocess, 'siglip2_model': siglip2_model.to(target_device), 
            'clap_tokenizer': clap_tokenizer, 'clap_model': clap_model.to(target_device), 
            'syncformer_preprocess': syncformer_preprocess,
            'syncformer_preprocess_tensor': syncformer_preprocess_tensor,
            'syncformer_model': syncformer_model.to(target_device)
        }
        return model_dict, cfg


class LoadDACHunyuanVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        vae_files = [f for f in folder_paths.get_filename_list("vae") if "dac" in f.lower() or "vae_128d_48k" in f.lower()]
        if os.path.exists(model_dir):
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file == "vae_128d_48k.pth":
                        rel_path = os.path.relpath(os.path.join(root, file), folder_paths.models_dir)
                        if rel_path not in vae_files:
                            vae_files.append(rel_path)
        # Always include the default path so download can be triggered even
        # before vae_128d_48k.pth has been downloaded.
        default_vae_rel = os.path.relpath(
            os.path.join(model_dir, "vae_128d_48k.pth"), folder_paths.models_dir
        )
        if default_vae_rel not in vae_files:
            vae_files.insert(0, default_vae_rel)
        if not vae_files:
            vae_files = ["hunyuan_foley/vae_128d_48k.pth"]
        return io.Schema(
            node_id="LoadDACHunyuanVAE",
            display_name="Hunyuan-Foley VAE loader",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Combo.Input("vae_name", options=vae_files)
            ],
            outputs=[
                io.Vae.Output()
            ]
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        global loaded_vaes_cpu
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        # Auto-download vae_128d_48k.pth if it's the selected file and is missing
        if "vae_128d_48k" in vae_name:
            ensure_vae_downloaded(model_dir)
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path:
            vae_path = os.path.join(folder_paths.models_dir, vae_name)
        if vae_path in loaded_vaes_cpu:
            return io.NodeOutput(loaded_vaes_cpu[vae_path])
        with suppress_output():
            vae = DAC.load(vae_path).eval()
        loaded_vaes_cpu[vae_path] = vae
        return io.NodeOutput(vae)


class HunyuanFoleySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HunyuanFoleySampler",
            display_name="Hunyuan-Foley Sampler",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Custom("FOLEY_MODEL").Input("foley_model"),
                io.Image.Input("video_frames"),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=1.0),
                io.String.Input("prompt", default="A person walks on frozen ice", multiline=True),
                io.String.Input("negative_prompt", default="noisy, harsh", multiline=True),
                io.Float.Input("guidance_scale", default=4.5, min=1.0, max=10.0, step=0.1),
                io.Int.Input("steps", default=50, min=10, max=100, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input(
                    "sync_batch_size", default=4, min=-1, max=64, step=1, optional=True,
                    tooltip=(
                        "Synchformer 每次处理的视频 segment 数量（仅 cpu_offload 模式下生效）。\n"
                        "每段 = 16帧，显存占用约 O(batch²)，推荐：\n"
                        "  4  → 适合 8 GB 显存，35秒以上视频\n"
                        "  8  → 适合 12 GB 显存，速度更快\n"
                        "  16 → 适合 24 GB 显存，最快\n"
                        "  -1 → 不分批（全部一次，适合短视频/大显存）"
                    )
                )
            ],
            outputs=[
                io.Latent.Output()
            ]
        )

    @classmethod
    def execute(cls, foley_model, video_frames, fps, prompt, negative_prompt,
                guidance_scale, steps, seed, sync_batch_size=4) -> io.NodeOutput:
        # Support both old 3-tuple and new 4-tuple (with cpu_offload flag)
        if len(foley_model) == 4:
            model_dict, cfg, precision, cpu_offload = foley_model
        else:
            model_dict, cfg, precision = foley_model
            cpu_offload = False

        # Auto-reload if models were unloaded by DACHunyuanVAEDecode
        if model_dict.get('__unloaded__'):
            reload_params = model_dict.get('__reload_params__')
            if reload_params is None:
                raise RuntimeError(
                    "Hunyuan-Foley models were unloaded and reload parameters are missing. "
                    "Please re-execute the 'Hunyuan-Foley Model Loader' node manually."
                )
            logging.info("CPU offload: models were unloaded — auto-reloading...")
            # Re-run the loader node (this repopulates loaded_models_cache and
            # returns a fresh tuple with a new model_dict)
            fresh_output = HunyuanFoleyModelLoader.execute(
                model_path_name=reload_params['model_path_name'],
                foley_checkpoint_name=reload_params['foley_checkpoint_name'],
                cpu_offload=reload_params['cpu_offload'],
            )
            # fresh_output is an io.NodeOutput whose .args[0] is the tuple
            fresh_tuple = fresh_output.args[0]
            if len(fresh_tuple) == 4:
                model_dict, cfg, precision, cpu_offload = fresh_tuple
            else:
                model_dict, cfg, precision = fresh_tuple
                cpu_offload = False
            logging.info("CPU offload: auto-reload complete.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        if not cpu_offload:
            # Original behaviour: move all models to GPU if needed
            current_device_type = next(model_dict['foley_model'].parameters()).device.type
            if current_device_type != device:
                logging.info(f"Moving models to '{device}'...")
                for model_name, model in model_dict.items():
                    if hasattr(model, 'to'):
                        if model_name == 'foley_model':
                            model_dict[model_name] = model.to(device, dtype=dtype)
                        else:
                            model_dict[model_name] = model.to(device)
                empty_cuda_cache()

        set_manual_seed(seed)
        
        num_frames = video_frames.shape[0]
        audio_len_in_s = num_frames / fps
        
        # --- SigLIP Processing ---
        siglip_fps = FPS_VISUAL["siglip2"]
        siglip_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * siglip_fps)).astype(int)
        
        frames_siglip_tensor = video_frames[siglip_indices]  # (B_sub, H, W, C)
        frames_siglip_np = (frames_siglip_tensor.cpu().numpy() * 255).astype(np.uint8)
        images_siglip_pil = [Image.fromarray(f).convert('RGB') for f in frames_siglip_np]

        if cpu_offload:
            # Flush any stale VRAM from previous ComfyUI nodes or incomplete
            # runs before we start moving models to GPU.  This is the primary
            # guard against intermittent OOMs on long videos.
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                gc.collect()
                torch.cuda.empty_cache()

            logging.info("CPU offload: moving SigLIP2 to GPU for feature extraction...")
            with model_on_device(model_dict['siglip2_model'], device):
                images_siglip = model_dict['siglip2_preprocess'](
                    images=images_siglip_pil, return_tensors="pt"
                ).to(device)
                # Process in small batches to cap activation VRAM for long videos
                _siglip_batch = 16
                _pooler_chunks = []
                with torch.no_grad():
                    for _i in range(0, images_siglip['pixel_values'].shape[0], _siglip_batch):
                        _batch_pv = images_siglip['pixel_values'][_i:_i + _siglip_batch]
                        _out = model_dict['siglip2_model'](pixel_values=_batch_pv)
                        _pooler_chunks.append(_out.pooler_output.cpu())
                        del _out
                # Free GPU tensors BEFORE context exits (model moves back to CPU)
                del images_siglip, _batch_pv
            siglip_feat = torch.cat(_pooler_chunks, dim=0).unsqueeze(0)
            del _pooler_chunks
            # Release SigLIP2 VRAM before Synchformer
            gc.collect()
            torch.cuda.empty_cache()
        else:
            images_siglip = model_dict['siglip2_preprocess'](images=images_siglip_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                siglip_output = model_dict['siglip2_model'](**images_siglip)
            siglip_feat = siglip_output.pooler_output.unsqueeze(0)

        # --- Synchformer Processing ---
        sync_fps = FPS_VISUAL["synchformer"]
        sync_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * sync_fps)).astype(int)
        
        frames_sync_tensor = video_frames[sync_indices].permute(0, 3, 1, 2)
        sync_frames = model_dict['syncformer_preprocess_tensor'](frames_sync_tensor).unsqueeze(0)

        if cpu_offload:
            logging.info("CPU offload: moving Synchformer to GPU for feature extraction...")
            with model_on_device(model_dict['syncformer_model'], device):
                model_dict_with_device = {**model_dict, 'device': device,
                                           'syncformer_model': model_dict['syncformer_model']}
                # Pass sync_frames as CPU tensor — encode_video_with_sync calls
                # .to(device) internally on only the stacked segments, avoiding
                # a duplicate full-video GPU copy (~1.35 GB for 35s video).
                # batch_size=4 limits Synchformer attention to 4 segments at a
                # time, preventing the O(n²) attention matrix from OOMing.
                sync_feat = encode_video_with_sync(
                    sync_frames, AttributeDict(model_dict_with_device), batch_size=sync_batch_size
                ).cpu()
                del model_dict_with_device
            del sync_frames
            # Release Synchformer VRAM before CLAP
            gc.collect()
            torch.cuda.empty_cache()
        else:
            sync_frames_dev = sync_frames.to(device)
            model_dict_with_device = {**model_dict, 'device': device}
            # batch_size=-1: process all segments at once (original behaviour;
            # fine when enough VRAM is available without cpu_offload)
            sync_feat = encode_video_with_sync(sync_frames_dev, AttributeDict(model_dict_with_device), batch_size=-1)

        # --- Text Feature Extraction ---
        prompts = [negative_prompt, prompt]

        if cpu_offload:
            logging.info("CPU offload: moving CLAP to GPU for text encoding...")
            with model_on_device(model_dict['clap_model'], device):
                clap_dict = {**model_dict, 'device': device,
                             'clap_model': model_dict['clap_model']}
                text_feat_res, _ = encode_text_feat(prompts, AttributeDict(clap_dict))
                del clap_dict
            text_feat_res = text_feat_res.cpu()
            # Release CLAP VRAM before denoising
            gc.collect()
            torch.cuda.empty_cache()
        else:
            model_dict_with_device = {**model_dict, 'device': device}
            text_feat_res, _ = encode_text_feat(prompts, AttributeDict(model_dict_with_device))

        text_feat, uncond_text_feat = text_feat_res[1:], text_feat_res[:1]
        if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
            text_feat = text_feat[:, :cfg.model_config.model_kwargs.text_length]
            uncond_text_feat = uncond_text_feat[:, :cfg.model_config.model_kwargs.text_length]
        
        logging.info(f"Generating audio ({audio_len_in_s:.2f}s)...")

        if cpu_offload:
            foley_model_obj = model_dict['foley_model']
            # Use block-level offload: each transformer block is moved to GPU
            # one at a time. Small auxiliary modules stay on GPU for efficiency.
            # See foley_block_offload_context() for details.
            with foley_block_offload_context(foley_model_obj, device, dtype):
                latents = denoise_process(
                    AttributeDict({'siglip2_feat': siglip_feat.to(device), 'syncformer_feat': sync_feat.to(device)}),
                    AttributeDict({'text_feat': text_feat.to(device), 'uncond_text_feat': uncond_text_feat.to(device)}),
                    audio_len_in_s,
                    AttributeDict({'foley_model': foley_model_obj, 'device': device}),
                    cfg, guidance_scale, steps
                )
        else:
            latents = denoise_process(
                AttributeDict({'siglip2_feat': siglip_feat, 'syncformer_feat': sync_feat}),
                AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat}),
                audio_len_in_s,
                AttributeDict({'foley_model': model_dict['foley_model'], 'device': device}),
                cfg, guidance_scale, steps
            )
        
        return io.NodeOutput({"samples": latents.cpu(), "audio_len_in_s": audio_len_in_s})


class DACHunyuanVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DACHunyuanVAEDecode",
            display_name="Hunyuan-Foley VAE Decode",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Boolean.Input(
                    "unload_models_after_use", default=False, optional=True,
                    tooltip="推理完成后将所有 Foley 模型彻底从内存（RAM/虚拟内存）中卸载，"
                            "释放 cpu_offload 模式下占用的系统内存。下次运行需重新加载。"
                ),
                io.Boolean.Input(
                    "unload_vae_after_use", default=False, optional=True,
                    tooltip="VAE 解码后同时从内存中卸载 VAE 模型。"
                )
            ],
            outputs=[
                io.Audio.Output()
            ]
        )

    @classmethod
    def execute(cls, samples, vae, unload_models_after_use=False,
                unload_vae_after_use=False) -> io.NodeOutput:
        latents = samples["samples"]
        audio_len_in_s = samples["audio_len_in_s"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vae_on_device = vae.to(device)
        try:
            with torch.no_grad():
                audio_tensor = vae_on_device.decode(latents.to(device)).float().cpu()
            sample_rate = vae.sample_rate
            audio_tensor = audio_tensor[:, :int(audio_len_in_s * sample_rate)]
            audio_out = {"waveform": audio_tensor, "sample_rate": sample_rate}
        finally:
            vae_on_device.to("cpu")
            empty_cuda_cache()

            if unload_models_after_use:
                unload_foley_models()

            if unload_vae_after_use:
                unload_vae_models()

        return io.NodeOutput(audio_out)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyModelLoader": HunyuanFoleyModelLoader,
    "LoadDACHunyuanVAE": LoadDACHunyuanVAE,
    "HunyuanFoleySampler": HunyuanFoleySampler,
    "DACHunyuanVAEDecode": DACHunyuanVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyModelLoader": "Hunyuan-Foley model loader",
    "LoadDACHunyuanVAE": "Hunyuan-Foley VAE loader",
    "HunyuanFoleySampler": "Hunyuan-Foley Sampler",
    "DACHunyuanVAEDecode": "Hunyuan-Foley VAE Decode",
}
