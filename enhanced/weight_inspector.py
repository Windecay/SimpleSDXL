import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class SafeTensorHeader:
    tensors: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


def _normalize_path_seps(s: str) -> str:
    return s.replace("\\", "/")


def _read_safetensors_header(path: str) -> SafeTensorHeader:
    file_size = os.path.getsize(path)
    if file_size < 9:
        raise ValueError("Invalid safetensors file size")
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError("Invalid safetensors file header")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        max_header_len = min(file_size - 8, 512 * 1024 * 1024)
        if header_len <= 0 or header_len > max_header_len:
            raise ValueError(f"Invalid safetensors header length: {header_len} (max {max_header_len})")
        header_json = f.read(header_len)
        header = json.loads(header_json)
    metadata = header.get("__metadata__", {}) or {}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return SafeTensorHeader(tensors=tensors, metadata=metadata)


def _read_u32_le(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise ValueError("Unexpected EOF")
    return struct.unpack("<I", b)[0]


def _read_u64_le(f) -> int:
    b = f.read(8)
    if len(b) != 8:
        raise ValueError("Unexpected EOF")
    return struct.unpack("<Q", b)[0]


def _read_gguf_string(f) -> str:
    n = _read_u64_le(f)
    if n > 256 * 1024 * 1024:
        raise ValueError(f"GGUF string too large: {n}")
    b = f.read(n)
    if len(b) != n:
        raise ValueError("Unexpected EOF")
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode(errors="replace")


def _read_gguf_value(f, value_type: int) -> Any:
    if value_type == 0:
        b = f.read(1)
        if len(b) != 1:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<B", b)[0]
    if value_type == 1:
        b = f.read(1)
        if len(b) != 1:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<b", b)[0]
    if value_type == 2:
        b = f.read(2)
        if len(b) != 2:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<H", b)[0]
    if value_type == 3:
        b = f.read(2)
        if len(b) != 2:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<h", b)[0]
    if value_type == 4:
        return _read_u32_le(f)
    if value_type == 5:
        b = f.read(4)
        if len(b) != 4:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<i", b)[0]
    if value_type == 6:
        b = f.read(4)
        if len(b) != 4:
            raise ValueError("Unexpected EOF")
        return struct.unpack("<f", b)[0]
    if value_type == 7:
        b = f.read(1)
        if len(b) != 1:
            raise ValueError("Unexpected EOF")
        return bool(struct.unpack("<B", b)[0])
    if value_type == 8:
        return _read_gguf_string(f)
    if value_type == 9:
        inner_type = _read_u32_le(f)
        n = _read_u64_le(f)
        if n > 10_000_000:
            raise ValueError(f"GGUF array too large: {n}")
        if inner_type == 8:
            return [_read_gguf_string(f) for _ in range(n)]
        if inner_type == 0:
            return [struct.unpack("<B", f.read(1))[0] for _ in range(n)]
        if inner_type == 1:
            return [struct.unpack("<b", f.read(1))[0] for _ in range(n)]
        if inner_type == 2:
            return [struct.unpack("<H", f.read(2))[0] for _ in range(n)]
        if inner_type == 3:
            return [struct.unpack("<h", f.read(2))[0] for _ in range(n)]
        if inner_type == 4:
            return [_read_u32_le(f) for _ in range(n)]
        if inner_type == 5:
            out = []
            for _ in range(n):
                b = f.read(4)
                if len(b) != 4:
                    raise ValueError("Unexpected EOF")
                out.append(struct.unpack("<i", b)[0])
            return out
        if inner_type == 6:
            out = []
            for _ in range(n):
                b = f.read(4)
                if len(b) != 4:
                    raise ValueError("Unexpected EOF")
                out.append(struct.unpack("<f", b)[0])
            return out
        if inner_type == 7:
            out = []
            for _ in range(n):
                b = f.read(1)
                if len(b) != 1:
                    raise ValueError("Unexpected EOF")
                out.append(bool(struct.unpack("<B", b)[0]))
            return out
        raise ValueError(f"Unsupported GGUF array inner type: {inner_type}")
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def _read_gguf_metadata_and_tensor_names(path: str) -> Tuple[Dict[str, Any], List[str], int]:
    p = os.path.abspath(path)
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = _read_u32_le(f)
        tensor_count = _read_u64_le(f)
        kv_count = _read_u64_le(f)
        metadata: Dict[str, Any] = {}
        for _ in range(kv_count):
            k = _read_gguf_string(f)
            t = _read_u32_le(f)
            v = _read_gguf_value(f, t)
            metadata[k] = v
        tensor_names: List[str] = []
        for _ in range(tensor_count):
            name = _read_gguf_string(f)
            tensor_names.append(name)
            n_dims = _read_u32_le(f)
            if n_dims > 32:
                raise ValueError(f"GGUF tensor dims too large: {n_dims}")
            for _ in range(n_dims):
                _read_u64_le(f)
            _read_u32_le(f)
            _read_u64_le(f)
        return metadata, tensor_names, int(version)


def _iter_tensor_shapes_from_safetensors(header: SafeTensorHeader) -> Iterable[Tuple[str, Tuple[int, ...]]]:
    for k, v in header.tensors.items():
        shape = v.get("shape")
        if isinstance(shape, list) and all(isinstance(x, int) for x in shape):
            yield k, tuple(shape)
        else:
            yield k, tuple()


def _is_lora_key(k: str) -> bool:
    kl = k.lower()
    if "lora_" in kl:
        return True
    if ".lora_" in kl:
        return True
    if kl.endswith(".lora_up.weight") or kl.endswith(".lora_down.weight"):
        return True
    if kl.endswith(".lora_up") or kl.endswith(".lora_down"):
        return True
    if kl.endswith(".lora_a.weight") or kl.endswith(".lora_b.weight"):
        return True
    if kl.endswith(".lora_a") or kl.endswith(".lora_b"):
        return True
    if "lycoris" in kl or "loha" in kl or "lokr" in kl:
        return True
    return False


def _lora_targets_from_keys(keys: Iterable[str]) -> List[str]:
    has_unet = False
    has_te = False
    has_te2 = False
    has_vae = False
    for k in keys:
        kl = k.lower()
        if not _is_lora_key(kl):
            continue
        if (
            "unet" in kl
            or "diffusion_model" in kl
            or "double_blocks." in kl
            or "single_blocks." in kl
            or "single_transformer_blocks." in kl
            or "transformer_blocks." in kl
            or "blocks." in kl
        ):
            has_unet = True
        if "text_encoder_2" in kl or "conditioner.embedders.1" in kl:
            has_te2 = True
        if (
            "text_encoder" in kl
            or "cond_stage_model" in kl
            or "conditioner.embedders." in kl
            or ".transformer.text_model." in kl
            or kl.startswith("clip_")
        ):
            has_te = True
        if "vae" in kl or "first_stage_model" in kl:
            has_vae = True
    targets: List[str] = []
    if has_unet:
        targets.append("unet")
    if has_te2:
        targets.append("text_encoder_2")
    if has_te:
        targets.append("text_encoder")
    if has_vae:
        targets.append("vae")
    if not targets:
        targets.append("unknown")
    return targets


def _parse_json_maybe(value: Any) -> Optional[Any]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or (not s.startswith("{") and not s.startswith("[")):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _infer_components_from_keys(keys: Iterable[str]) -> List[str]:
    has_unet = False
    has_te = False
    has_te2 = False
    has_vae = False
    for k in keys:
        kl = k.lower()
        if (
            "unet" in kl
            or "diffusion_model" in kl
            or "double_blocks." in kl
            or "single_blocks." in kl
            or "single_transformer_blocks." in kl
            or "transformer_blocks." in kl
            or "blocks." in kl
        ):
            has_unet = True
        if "conditioner.embedders.1" in kl or "text_encoder_2" in kl:
            has_te2 = True
        if (
            "conditioner.embedders.0" in kl
            or "conditioner.embedders." in kl
            or "cond_stage_model" in kl
            or "text_encoder" in kl
            or ".transformer.text_model." in kl
            or kl.startswith("clip_")
        ):
            has_te = True
        if "first_stage_model" in kl or kl.startswith("vae.") or kl.startswith("encoder.") or kl.startswith("decoder."):
            has_vae = True
    out: List[str] = []
    if has_unet:
        out.append("unet")
    if has_te2:
        out.append("text_encoder_2")
    if has_te:
        out.append("text_encoder")
    if has_vae:
        out.append("vae")
    if not out:
        out.append("unknown")
    return out


def _make_key_signature(keys: List[str]) -> Dict[str, Any]:
    sample = keys[: min(len(keys), 10000)]
    prefix1: Dict[str, int] = {}
    prefix2: Dict[str, int] = {}
    patterns = {
        "has_model_diffusion_model": False,
        "has_conditioner_embedders": False,
        "has_cond_stage_model": False,
        "has_double_blocks": False,
        "has_single_blocks": False,
        "has_single_transformer_blocks": False,
        "has_transformer_blocks": False,
        "has_cross_attn": False,
        "has_k_img_or_v_img": False,
        "has_cap_embedder": False,
        "has_context_refiner": False,
    }

    for k in sample:
        kl = k.lower()
        parts = k.split(".")
        if parts:
            p1 = parts[0]
            prefix1[p1] = prefix1.get(p1, 0) + 1
        if len(parts) >= 2:
            p2 = f"{parts[0]}.{parts[1]}"
            prefix2[p2] = prefix2.get(p2, 0) + 1

        if "model.diffusion_model" in kl:
            patterns["has_model_diffusion_model"] = True
        if "conditioner.embedders" in kl:
            patterns["has_conditioner_embedders"] = True
        if "cond_stage_model" in kl:
            patterns["has_cond_stage_model"] = True
        if "double_blocks." in kl:
            patterns["has_double_blocks"] = True
        if "single_blocks." in kl:
            patterns["has_single_blocks"] = True
        if "single_transformer_blocks." in kl:
            patterns["has_single_transformer_blocks"] = True
        if "transformer_blocks." in kl:
            patterns["has_transformer_blocks"] = True
        if "cross_attn" in kl:
            patterns["has_cross_attn"] = True
        if "k_img" in kl or "v_img" in kl:
            patterns["has_k_img_or_v_img"] = True
        if "cap_embedder" in kl:
            patterns["has_cap_embedder"] = True
        if "context_refiner" in kl:
            patterns["has_context_refiner"] = True

    def top_items(d: Dict[str, int], limit: int) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda x: (-x[1], x[0]))[:limit]

    return {
        "sample_key_count": len(sample),
        "prefix1_top": top_items(prefix1, 30),
        "prefix2_top": top_items(prefix2, 30),
        "patterns": patterns,
    }


def _infer_arch_family_from_keys(keys: List[str], metadata: Dict[str, Any]) -> str:
    hint_arch: Optional[str] = None
    for v in (metadata or {}).values():
        try:
            s = str(v).lower()
        except Exception:
            continue
        if "sdpose" in s:
            return "sdpose"
        if "qwen" in s:
            return "qwen"
        if "z-image" in s or "z_image" in s or "zimage" in s:
            return "z_image"
        if "flux" in s:
            return "flux"
        if "wan" in s or "wan2" in s:
            return "wan"
        if hint_arch is None and ("sdxl" in s or "sd-xl" in s or "stable-diffusion-xl" in s):
            hint_arch = "sdxl"
        if hint_arch is None and ("sd3" in s or "sd-3" in s):
            hint_arch = "sd3"
        if hint_arch is None and ("sd15" in s or "sd1.5" in s or "sd_1.5" in s or "sd-v1-5" in s):
            hint_arch = "sdxl"

    arch = str(metadata.get("modelspec.architecture", "") or "")
    if arch:
        arch_l = arch.lower()
        if "sd-3" in arch_l or "sd3" in arch_l:
            hint_arch = hint_arch or "sd3"
        if (
            "sd-xl" in arch_l
            or "sdxl" in arch_l
            or "stable-diffusion-xl" in arch_l
            or re.search(r"(diffusion|stable)[-_ ]?diffusion[-_ ]?xl", arch_l)
            or "xl-v1" in arch_l
        ):
            hint_arch = hint_arch or "sdxl"
        if "sd-1" in arch_l or "sd1" in arch_l or "sd-v1" in arch_l:
            hint_arch = hint_arch or "sdxl"
        if "flux" in arch_l:
            return "flux"
        if "wan" in arch_l:
            return "wan"

    model_class = str(metadata.get("model_class", "") or "")
    if model_class:
        mc_l = model_class.lower()
        if "flux" in mc_l:
            return "flux"
        if "qwen" in mc_l:
            return "qwen"
        if "wan" in mc_l:
            return "wan"

    config_obj = _parse_json_maybe(metadata.get("config"))
    if isinstance(config_obj, dict):
        cn = str(config_obj.get("_class_name", "") or "").lower()
        if "flux" in cn:
            return "flux"
        if "qwen" in cn:
            return "qwen"
        if "wan" in cn:
            return "wan"

    comfy_obj = _parse_json_maybe(metadata.get("comfy_config"))
    if isinstance(comfy_obj, dict):
        mc = str(comfy_obj.get("model_class", "") or "").lower()
        if "flux" in mc:
            return "flux"
        if "qwen" in mc:
            return "qwen"
        if "wan" in mc:
            return "wan"
        if "sdpose" in mc:
            return "sdpose"

    md_base = str(metadata.get("ss_base_model_version", "") or "")
    if md_base:
        md_l = md_base.lower()
        if "sdxl" in md_l:
            hint_arch = hint_arch or "sdxl"
        if "sd3" in md_l:
            hint_arch = hint_arch or "sd3"
        if "sd-v2" in md_l or "sd2" in md_l:
            hint_arch = hint_arch or "sd2"
        if "sd_1.5" in md_l or "sd-v1-5" in md_l or "sd15" in md_l or "sd1.5" in md_l:
            hint_arch = hint_arch or "sdxl"
        if "flux" in md_l:
            return "flux"
        if "wan" in md_l:
            return "wan"

    joined = "\n".join(keys[: min(len(keys), 5000)]).lower()
    if "llm_adapter.blocks.0.cross_attn.q_proj.weight" in joined and "blocks.0.mlp.layer1.weight" in joined:
        return "anima"
    if "mask_estimators." in joined and "band_split." in joined and "layers." in joined:
        return "melband_roformer"

    if "double_blocks." in joined and "img_attn" in joined and "txt_attn" in joined:
        return "flux"
    if "single_transformer_blocks." in joined:
        return "flux"
    if "lora_unet_double_blocks_" in joined or "lora_unet_single_transformer_blocks_" in joined:
        return "flux"
    if (
        "time_text_embed" in joined
        and ("x_embedder" in joined or "context_embedder" in joined)
        and ("transformer_blocks" in joined or "single_transformer_blocks" in joined)
    ):
        return "flux"
    if (
        "lora_unet_transformer_blocks_" in joined
        and "attn_add_" in joined
        and ("img_mlp_net" in joined or "img_mlp.net" in joined)
        and ("txt_mlp_net" in joined or "txt_mlp.net" in joined)
        and "img_mod" in joined
        and "txt_mod" in joined
    ):
        return "qwen"
    if (
        "transformer_blocks" in joined
        and "img_mlp_net" in joined
        and "txt_mlp_net" in joined
        and "img_mod" in joined
        and "txt_mod" in joined
        and "attn_add_" in joined
    ):
        return "flux"
    if (
        "transformer_blocks." in joined
        and "attn.add_k_proj" in joined
        and "img_mlp.net" in joined
        and "img_mod" in joined
        and "txt_mod" in joined
    ):
        return "qwen"
    if "vocoder." in joined and ("audio_vae." in joined or "audio_vae_" in joined or "\naudio_vae" in joined):
        return "ltx2"
    if "lora_te2_" in joined or "lora_te2." in joined:
        return "sdxl"
    if "text_encoder_2" in joined or "conditioner.embedders.1" in joined:
        return "sdxl"
    if "model.diffusion_model" in joined and "cond_stage_model" in joined:
        if "cond_stage_model.model" in joined:
            return "sd2"
        if "text_encoder_2" in joined or "conditioner.embedders." in joined:
            return "sdxl"
        return "sdxl"
    if "qwen" in joined or ("transformer_blocks." in joined and "time_text_embed" in joined and "img_in" in joined):
        return "qwen"
    if "transformer.transformer_blocks" in joined:
        if (
            "attn.add_q_proj" in joined
            or "attn.add_k_proj" in joined
            or "attn.add_v_proj" in joined
            or "attn.add_out_proj" in joined
            or "attn_add_q_proj" in joined
            or "attn_add_k_proj" in joined
            or "attn_add_v_proj" in joined
            or "attn_add_out_proj" in joined
        ):
            return "qwen"
        return "sd3"
    if "lora_unet_input_blocks_" in joined or "lora_unet_output_blocks_" in joined:
        return "sdxl"
    if "lora_unet_down_blocks_3" in joined or "lora_unet_up_blocks_3" in joined:
        return "sdxl"
    if "lora_unet_add_embedding" in joined or "lora_unet_add_time_embedding" in joined:
        return "sdxl"
    if (
        "lora_unet_down_blocks_" in joined
        and "lora_unet_down_blocks_3" not in joined
        and "lora_unet_up_blocks_3" not in joined
        and ("lora_unet_down_blocks_2" in joined or "lora_unet_up_blocks_2" in joined)
    ):
        return "sdxl"
    if (
        "diffusion_model.blocks." in joined
        and "cross_attn.k" in joined
        and "cross_attn.q" in joined
        and "cross_attn.v" in joined
        and "self_attn.q" in joined
        and "self_attn.k" in joined
        and "self_attn.v" in joined
    ):
        return "wan"
    if (
        "transformer_blocks." in joined
        and ("attn.add_k_proj" in joined or "attn_add_k_proj" in joined)
        and "img_mlp" in joined
        and ("txt_mlp" in joined or "txt_mod" in joined)
    ):
        return "qwen"
    if (
        "model.diffusion_model.blocks." in joined
        and "self_attn" in joined
        and "cross_attn" in joined
        and "patch_embedding" in joined
        and "text_embedding" in joined
    ):
        return "wan"
    if "cross_attn" in joined and ("k_img" in joined or "v_img" in joined):
        return "wan"
    if "wan" in joined or "wan2" in joined:
        return "wan"
    if "cap_embedder" in joined and "context_refiner" in joined:
        return "z_image"
    return hint_arch or "unknown"


def _infer_arch_family_from_filename(path: str) -> str:
    p = _normalize_path_seps(os.path.abspath(path))
    base = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    s = f"{parent} {base}"

    if "sdpose" in s:
        return "sdpose"
    if "ltx2" in s or ("ltx" in s and "2" in s):
        return "ltx2"
    if "newbie" in s:
        return "newbie"
    if "z_image" in s or "z-image" in s or "zimage" in s or "zit" in s or "zib" in s:
        return "z_image"
    if "qwen" in s:
        return "qwen"
    if "wan" in s or re.search(r"(^|[^a-z0-9])wan([^a-z0-9]|$)", s):
        return "wan"
    if "kontext" in s:
        return "flux"
    if "flux" in s or "f.1" in s or "flux2" in s or "f.2" in s or "klein" in s:
        return "flux"
    if "sdxl" in s or "sd-xl" in s or "xl" in s or re.search(r"(^|[^a-z0-9])xl([^a-z0-9]|$)", s):
        return "sdxl"
    if "sd15" in s or "sd1.5" in s or "sd_1.5" in s or "v1-5" in s or "sd-v1-5" in s:
        return "sdxl"
    return "unknown"

def _infer_weight_kind(keys: List[str], metadata: Dict[str, Any]) -> str:
    if any(_is_lora_key(k) for k in keys):
        return "lora"
    arch = str(metadata.get("modelspec.architecture", "") or "").lower()
    if "textual-inversion" in arch or "embedding" in arch:
        return "embedding"
    if any(k.startswith("emb_params") for k in keys):
        return "embedding"
    if len(keys) <= 10 and all(k.lower().startswith("clip_") for k in keys):
        return "embedding"
    if any(k.lower().startswith("first_stage_model.") or k.lower().startswith("vae.") for k in keys):
        has_unet = any("diffusion_model" in k.lower() or k.lower().startswith("unet.") for k in keys)
        if not has_unet and len(keys) > 50:
            return "vae"
    if metadata.get("ss_network_module") or metadata.get("ss_network_dim"):
        return "lora"
    return "checkpoint"


def _read_safetensors_keys_metadata_and_shapes(path: str) -> Tuple[List[str], Dict[str, Any], List[Tuple[str, Tuple[int, ...]]], str]:
    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            metadata = f.metadata() or {}
            shapes: List[Tuple[str, Tuple[int, ...]]] = []
            for k in keys:
                kl = k.lower()
                if not (_is_lora_key(kl) or "lora" in kl):
                    continue
                if not (
                    kl.endswith(".lora_down.weight")
                    or kl.endswith(".lora_up.weight")
                    or kl.endswith(".lora_a.weight")
                    or kl.endswith(".lora_b.weight")
                    or kl.endswith(".lora_down")
                    or kl.endswith(".lora_up")
                    or kl.endswith(".lora_a")
                    or kl.endswith(".lora_b")
                ):
                    continue
                try:
                    t = f.get_tensor(k)
                    shape = tuple(int(x) for x in getattr(t, "shape", ()))
                    shapes.append((k, shape))
                except Exception:
                    continue
        keys.sort()
        return keys, metadata, shapes, "safe_open"
    except Exception:
        header = _read_safetensors_header(path)
        keys = list(header.tensors.keys())
        keys.sort()
        shapes = list(_iter_tensor_shapes_from_safetensors(header))
        return keys, header.metadata, shapes, "header"


def _infer_lora_rank_from_shapes(shapes: Iterable[Tuple[str, Tuple[int, ...]]]) -> Dict[str, Any]:
    ranks: List[int] = []
    for k, shape in shapes:
        kl = k.lower()
        if not (_is_lora_key(kl) or "lora" in kl):
            continue
        if not shape:
            continue
        if (kl.endswith(".lora_down.weight") or kl.endswith(".lora_down")) and len(shape) >= 2:
            ranks.append(int(shape[0]))
        elif (kl.endswith(".lora_a.weight") or kl.endswith(".lora_b.weight") or kl.endswith(".lora_a") or kl.endswith(".lora_b")) and len(shape) >= 2:
            ranks.append(int(min(shape[0], shape[1])))
    ranks = [r for r in ranks if r > 0]
    if not ranks:
        return {}
    ranks_sorted = sorted(ranks)
    return {
        "rank_min": ranks_sorted[0],
        "rank_max": ranks_sorted[-1],
        "rank_median": ranks_sorted[len(ranks_sorted) // 2],
        "rank_unique": sorted(set(ranks_sorted))[:32],
        "rank_count": len(ranks_sorted),
    }


def inspect_weight_file(
    path: str,
    torch_ckpt_load: bool = False,
    include_metadata: bool = True,
    include_key_examples: bool = True,
) -> Dict[str, Any]:
    path_abs = os.path.abspath(path)
    ext = os.path.splitext(path_abs)[1].lower()
    result: Dict[str, Any] = {
        "path": path_abs,
        "ext": ext,
        "file_type": "unknown",
    }

    if ext == ".safetensors":
        keys, metadata, shapes, parse_mode = _read_safetensors_keys_metadata_and_shapes(path_abs)
        lora_key_count = sum(1 for k in keys if _is_lora_key(k))
        components = _infer_components_from_keys(keys)
        signature = _make_key_signature(keys)
        arch_family = _infer_arch_family_from_keys(keys, metadata)
        arch_family_from_filename = _infer_arch_family_from_filename(path_abs)
        if arch_family_from_filename == "newbie":
            arch_family = "newbie"
        elif arch_family == "unknown" and arch_family_from_filename != "unknown":
            arch_family = arch_family_from_filename
        out_metadata = metadata if include_metadata else {}
        result.update(
            {
                "file_type": "safetensors",
                "parse_mode": parse_mode,
                "metadata": out_metadata,
                "key_count": len(keys),
                "lora_key_count": lora_key_count,
                "weight_kind": _infer_weight_kind(keys, out_metadata),
                "arch_family": arch_family,
                "lora_targets": _lora_targets_from_keys(keys) if lora_key_count else [],
                "components": components,
                "signature": signature,
                "lora_rank": _infer_lora_rank_from_shapes(shapes),
                "key_examples": (
                    {
                        "lora": [k for k in keys if _is_lora_key(k)][:20],
                        "top": keys[:20],
                    }
                    if include_key_examples
                    else {}
                ),
            }
        )
        return result

    if ext in (".ckpt", ".pt", ".pth", ".bin"):
        result["file_type"] = "torch"
        if not torch_ckpt_load:
            result["note"] = "torch_ckpt_load=false; skipped loading"
            return result
        try:
            import torch
        except Exception as e:
            result["error"] = f"torch import failed: {type(e).__name__}: {e}"
            return result
        obj = torch.load(path_abs, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif isinstance(obj, dict):
            state_dict = obj
        else:
            result["note"] = f"unsupported torch object: {type(obj).__name__}"
            return result
        keys = sorted(map(str, state_dict.keys()))
        lora_key_count = sum(1 for k in keys if _is_lora_key(k))
        components = _infer_components_from_keys(keys)
        signature = _make_key_signature(keys)
        arch_family = _infer_arch_family_from_keys(keys, {})
        arch_family_from_filename = _infer_arch_family_from_filename(path_abs)
        if arch_family_from_filename == "newbie":
            arch_family = "newbie"
        elif arch_family == "unknown" and arch_family_from_filename != "unknown":
            arch_family = arch_family_from_filename
        shapes: List[Tuple[str, Tuple[int, ...]]] = []
        for k in keys[: min(2000, len(keys))]:
            v = state_dict.get(k)
            shape = tuple(getattr(v, "shape", ())) if v is not None else tuple()
            shapes.append((k, tuple(int(x) for x in shape) if shape else tuple()))
        result.update(
            {
                "key_count": len(keys),
                "lora_key_count": lora_key_count,
                "weight_kind": _infer_weight_kind(keys, {}),
                "arch_family": arch_family,
                "lora_targets": _lora_targets_from_keys(keys) if lora_key_count else [],
                "components": components,
                "signature": signature,
                "lora_rank": _infer_lora_rank_from_shapes(shapes),
                "key_examples": (
                    {
                        "lora": [k for k in keys if _is_lora_key(k)][:20],
                        "top": keys[:20],
                    }
                    if include_key_examples
                    else {}
                ),
            }
        )
        return result

    if ext == ".gguf":
        result["file_type"] = "gguf"
        result["parse_mode"] = "gguf"
        try:
            gguf_meta, gguf_tensors, gguf_ver = _read_gguf_metadata_and_tensor_names(path_abs)
            keys = sorted(map(str, gguf_tensors))
            out_metadata = {f"gguf.{k}": v for k, v in (gguf_meta or {}).items()}
            out_metadata["gguf.version"] = gguf_ver
            lora_key_count = sum(1 for k in keys if _is_lora_key(k))
            components = _infer_components_from_keys(keys)
            signature = _make_key_signature(keys)
            arch_family = _infer_arch_family_from_keys(keys, out_metadata)
            arch_family_from_filename = _infer_arch_family_from_filename(path_abs)
            if arch_family == "unknown" and arch_family_from_filename != "unknown":
                arch_family = arch_family_from_filename
            result.update(
                {
                    "metadata": {} if not include_metadata else out_metadata,
                    "key_count": len(keys),
                    "lora_key_count": lora_key_count,
                    "weight_kind": _infer_weight_kind(keys, out_metadata),
                    "arch_family": arch_family,
                    "lora_targets": _lora_targets_from_keys(keys) if lora_key_count else [],
                    "components": components,
                    "signature": signature,
                    "lora_rank": {},
                    "key_examples": (
                        {
                            "lora": [k for k in keys if _is_lora_key(k)][:20],
                            "top": keys[:20],
                        }
                        if include_key_examples
                        else {}
                    ),
                }
            )
        except Exception as e:
            result["weight_kind"] = "checkpoint"
            result["arch_family"] = _infer_arch_family_from_filename(path_abs)
            if result["arch_family"] == "sd15":
                result["arch_family"] = "sdxl"
            result["error"] = f"GGUF parse failed: {type(e).__name__}: {e}"
        return result

    result["note"] = "unsupported file extension"
    return result
