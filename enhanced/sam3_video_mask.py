import json
import importlib.util
import logging
import os
import sys
import tempfile
import uuid

import cv2
import numpy as np
import torch
from contextlib import nullcontext


_VIDEO_PREDICTOR = None
_VIDEO_PREDICTOR_CKPT = None
logger = logging.getLogger(__name__)
SAM3_PT_URL = "https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_easy_sam3_importable() -> None:
    easy_sam3_root = os.path.join(
        _repo_root(), "comfy", "custom_nodes", "ComfyUI-Easy-Sam3"
    )
    if easy_sam3_root not in sys.path:
        sys.path.insert(0, easy_sam3_root)


def _bytes_to_gib(n: int) -> float:
    return float(n) / (1024.0**3)


def _gib_to_bytes(gib: float) -> int:
    return int(float(gib) * (1024.0**3))


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("0", "false", "no", "off", ""):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return bool(default)


def _apply_elastic_vram_limit_for_sam3(*, device_index: int) -> dict | None:
    if not torch.cuda.is_available():
        return None
    if not _env_flag("SIMPLEAI_SAM3_ELASTIC_VRAM", True):
        return None

    try:
        torch.cuda.set_device(int(device_index))
    except Exception:
        pass

    headroom_gb = _env_float("SIMPLEAI_SAM3_NVML_HEADROOM_GB", 1.0)
    limit_gb = _env_float("SIMPLEAI_SAM3_LIMIT_GB", 0.0)

    try:
        from ldm_patched.modules.model_management import get_free_memory_by_nvml_for_nvidia
    except Exception:
        return None

    dev = None
    try:
        dev = torch.device(f"cuda:{int(device_index)}")
    except Exception:
        dev = None

    free_bytes = None
    try:
        free_bytes = get_free_memory_by_nvml_for_nvidia(dev)
    except Exception:
        free_bytes = None
    if free_bytes is None:
        return None

    try:
        total_bytes = int(torch.cuda.get_device_properties(int(device_index)).total_memory)
    except Exception:
        total_bytes = 0
    if total_bytes <= 0:
        return None

    pid_used_bytes = 0
    try:
        pid_used_bytes = int(torch.cuda.memory_reserved(int(device_index)))
    except Exception:
        pid_used_bytes = 0

    headroom_bytes = _gib_to_bytes(float(headroom_gb))
    cap_from_free_bytes = int(pid_used_bytes + max(0, int(free_bytes) - headroom_bytes))

    cap_bytes = cap_from_free_bytes
    if float(limit_gb) and float(limit_gb) > 0:
        cap_bytes = min(int(cap_bytes), _gib_to_bytes(float(limit_gb)))

    frac = float(cap_bytes) / float(total_bytes)
    frac = max(0.01, min(1.0, frac))
    try:
        torch.cuda.set_per_process_memory_fraction(float(frac), device=int(device_index))
    except Exception:
        return None

    return {
        "enabled": True,
        "device_index": int(device_index),
        "limit_gb": float(limit_gb),
        "headroom_gb": float(headroom_gb),
        "cap_gb_effective": round(_bytes_to_gib(int(cap_bytes)), 3),
        "cap_gb_from_free": round(_bytes_to_gib(int(cap_from_free_bytes)), 3),
        "fraction": round(float(frac), 6),
    }


def _resolve_sam3_checkpoint(model_path: str | None) -> str:
    import modules.config as config
    from modules.model_loader import load_file_from_url

    if model_path:
        model_path = os.path.abspath(str(model_path))
        if os.path.isfile(model_path):
            return model_path

    models_root = config.get_path_models_root()
    model_dir = os.path.abspath(os.path.join(models_root, "sam3"))
    ckpt_path = os.path.abspath(os.path.join(model_dir, "sam3.pt"))
    if not os.path.isfile(ckpt_path):
        load_file_from_url(url=SAM3_PT_URL, model_dir=model_dir, file_name="sam3.pt")
    if os.path.isfile(ckpt_path):
        return ckpt_path
    raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt_path}")


def _get_video_meta(video_path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frame_count, width, height


def _is_video_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path)[-1].lower()
    return ext in [".mp4", ".mov", ".mkv", ".avi", ".webm"]


def _extract_video_to_temp_frames(video_path: str, *, max_frames: int) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        if int(max_frames) >= 0 and idx >= int(max_frames):
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out_path = os.path.join(tmp_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        idx += 1

    cap.release()
    if idx <= 0:
        raise RuntimeError("No frames extracted from video.")
    return tmp_dir


def _parse_editor_payload(payload_json: str) -> dict:
    if not payload_json or not str(payload_json).strip():
        return {}
    try:
        data = json.loads(payload_json)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_points(points: list, width: int, height: int) -> list[list[float]]:
    out: list[list[float]] = []
    if not points:
        return out
    for p in points:
        if isinstance(p, dict):
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(p[0])
            y = float(p[1])
        else:
            continue

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append([x, y])
            continue

        if width > 0 and height > 0:
            out.append([max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))])
    return out


def _normalize_bbox(bbox: object, width: int, height: int) -> list[list[float]] | None:
    if not bbox:
        return None
    try:
        if isinstance(bbox, dict):
            x0 = float(bbox.get("x0", bbox.get("x", 0.0)))
            y0 = float(bbox.get("y0", bbox.get("y", 0.0)))
            x1 = float(bbox.get("x1", bbox.get("x2", 0.0)))
            y1 = float(bbox.get("y1", bbox.get("y2", 0.0)))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            x1 = float(bbox[2])
            y1 = float(bbox[3])
        else:
            return None

        if 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
        elif width > 0 and height > 0:
            nx0 = max(0.0, min(1.0, x0 / width))
            ny0 = max(0.0, min(1.0, y0 / height))
            nx1 = max(0.0, min(1.0, x1 / width))
            ny1 = max(0.0, min(1.0, y1 / height))
        else:
            return None

        xmin = min(nx0, nx1)
        ymin = min(ny0, ny1)
        w = abs(nx1 - nx0)
        h = abs(ny1 - ny0)
        if w <= 0.0 or h <= 0.0:
            return None
        return [[xmin, ymin, w, h]]
    except Exception:
        return None


def _select_binary_mask_from_outputs(
    outputs: dict, preferred_obj_id: int | None = None
) -> np.ndarray | None:
    mask = outputs.get("out_binary_masks", None)
    if mask is None or not isinstance(mask, np.ndarray):
        return None

    if mask.ndim == 2:
        return mask.astype(bool)

    if mask.ndim != 3 or mask.shape[0] <= 0:
        return None

    if preferred_obj_id is not None:
        obj_ids = outputs.get("out_obj_ids", None)
        try:
            if isinstance(obj_ids, np.ndarray):
                obj_ids_list = obj_ids.tolist()
            elif isinstance(obj_ids, (list, tuple)):
                obj_ids_list = list(obj_ids)
            else:
                obj_ids_list = []
            if preferred_obj_id in obj_ids_list:
                idx = int(obj_ids_list.index(preferred_obj_id))
                if 0 <= idx < int(mask.shape[0]):
                    return mask[idx].astype(bool)
        except Exception:
            pass

    probs = outputs.get("out_probs", None)
    if isinstance(probs, np.ndarray) and probs.ndim == 1 and int(probs.shape[0]) == int(mask.shape[0]):
        try:
            idx = int(np.argmax(probs))
            return mask[idx].astype(bool)
        except Exception:
            pass

    return np.any(mask, axis=0)


def _load_video_predictor(checkpoint_path: str):
    global _VIDEO_PREDICTOR, _VIDEO_PREDICTOR_CKPT

    if _VIDEO_PREDICTOR is not None and _VIDEO_PREDICTOR_CKPT == checkpoint_path:
        try:
            _apply_elastic_vram_limit_for_sam3(device_index=int(torch.cuda.current_device()))
        except Exception:
            pass
        _patch_predictor_for_webui(_VIDEO_PREDICTOR)
        return _VIDEO_PREDICTOR

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video predictor requires CUDA (torch.cuda.is_available() is False).")

    _ensure_easy_sam3_importable()
    from sam3.model_builder import build_sam3_video_predictor  # noqa: E402

    try:
        _apply_elastic_vram_limit_for_sam3(device_index=int(torch.cuda.current_device()))
    except Exception:
        pass
    _VIDEO_PREDICTOR = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
    _patch_predictor_for_webui(_VIDEO_PREDICTOR)
    _VIDEO_PREDICTOR_CKPT = checkpoint_path
    return _VIDEO_PREDICTOR


def unload_sam3_video_predictor() -> None:
    global _VIDEO_PREDICTOR, _VIDEO_PREDICTOR_CKPT

    predictor = _VIDEO_PREDICTOR
    _VIDEO_PREDICTOR = None
    _VIDEO_PREDICTOR_CKPT = None

    try:
        model = getattr(predictor, "model", None) if predictor is not None else None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        try:
            if predictor is not None and hasattr(predictor, "model"):
                predictor.model = None
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
    finally:
        try:
            del predictor
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _patch_predictor_for_webui(predictor) -> None:
    model = getattr(predictor, "model", None)
    if model is None:
        return
    if getattr(model, "_simpai_webui_patched", False):
        return

    _sync_bias_dtype_with_weight(model)
    try:
        if callable(getattr(model, "init_state", None)) and not getattr(
            model, "_simpai_init_state_offload_patched", False
        ):
            orig_init_state = model.init_state

            def init_state_patched(*args, **kwargs):
                if "offload_video_to_cpu" not in kwargs:
                    kwargs["offload_video_to_cpu"] = True
                return orig_init_state(*args, **kwargs)

            model.init_state = init_state_patched
            model._simpai_init_state_offload_patched = True
    except Exception:
        pass
    try:
        if callable(getattr(model, "_init_new_tracker_state", None)) and not getattr(
            model, "_simpai_init_new_tracker_state_patched", False
        ):
            orig_init_new_tracker_state = model._init_new_tracker_state

            def _init_new_tracker_state_patched(inference_state):
                offload_state_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_STATE_TO_CPU", True)
                tracker = getattr(model, "tracker", None)
                if tracker is not None and callable(getattr(tracker, "init_state", None)):
                    return tracker.init_state(
                        cached_features=inference_state["feature_cache"],
                        video_height=inference_state["orig_height"],
                        video_width=inference_state["orig_width"],
                        num_frames=inference_state["num_frames"],
                        offload_state_to_cpu=bool(offload_state_to_cpu),
                    )
                return orig_init_new_tracker_state(inference_state)

            model._init_new_tracker_state = _init_new_tracker_state_patched
            model._simpai_init_new_tracker_state_patched = True
    except Exception:
        pass
    try:
        cls = model.__class__
        if callable(getattr(cls, "_cache_frame_outputs", None)) and not getattr(
            cls, "_simpai_cache_frame_outputs_patched", False
        ):
            orig_cache_frame_outputs = cls._cache_frame_outputs

            def _cache_frame_outputs_patched(
                self,
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=None,
                removed_obj_ids=None,
                unconfirmed_obj_ids=None,
            ):
                offload_cached_masks = _env_flag(
                    "SIMPLEAI_SAM3_OFFLOAD_CACHED_MASKS_TO_CPU", True
                )
                if offload_cached_masks and isinstance(obj_id_to_mask, dict):
                    try:
                        converted = {}
                        for k, v in obj_id_to_mask.items():
                            if isinstance(v, torch.Tensor) and v.is_cuda:
                                converted[k] = v.detach().to("cpu", non_blocking=True)
                            else:
                                converted[k] = v
                        obj_id_to_mask = converted
                    except Exception:
                        pass
                return orig_cache_frame_outputs(
                    self,
                    inference_state,
                    frame_idx,
                    obj_id_to_mask,
                    suppressed_obj_ids=suppressed_obj_ids,
                    removed_obj_ids=removed_obj_ids,
                    unconfirmed_obj_ids=unconfirmed_obj_ids,
                )

            cls._cache_frame_outputs = _cache_frame_outputs_patched
            cls._simpai_cache_frame_outputs_patched = True
    except Exception:
        pass
    try:
        tracker = getattr(model, "tracker", None)
        tracker_cls = tracker.__class__ if tracker is not None else None
        if (
            tracker_cls is not None
            and callable(getattr(tracker_cls, "propagate_in_video", None))
            and not getattr(tracker_cls, "_simpai_prune_noncond_patched", False)
        ):
            orig_propagate_in_video = tracker_cls.propagate_in_video

            def propagate_in_video_patched(
                self,
                inference_state,
                start_frame_idx,
                max_frame_num_to_track,
                reverse,
                tqdm_disable=False,
                obj_ids=None,
                run_mem_encoder=True,
                propagate_preflight=False,
            ):
                enabled = _env_flag("SIMPLEAI_SAM3_PRUNE_NONCOND_FRAMES", True)
                keep_frames_env = int(_env_float("SIMPLEAI_SAM3_NONCOND_KEEP_FRAMES", 0.0))
                keep_frames_default = int(getattr(self, "max_obj_ptrs_in_encoder", 16) or 16) + 4
                keep_frames = keep_frames_env if keep_frames_env > 0 else keep_frames_default

                def _prune(state, current_frame_idx: int, is_reverse: bool):
                    if not enabled:
                        return
                    output_dict = state.get("output_dict", None) if isinstance(state, dict) else None
                    if not isinstance(output_dict, dict):
                        return
                    non_cond = output_dict.get("non_cond_frame_outputs", None)
                    if not isinstance(non_cond, dict) or not non_cond:
                        return
                    consolidated = state.get("consolidated_frame_inds", {}) if isinstance(state, dict) else {}
                    consolidated_non_cond = (
                        consolidated.get("non_cond_frame_outputs", set())
                        if isinstance(consolidated, dict)
                        else set()
                    )
                    if is_reverse:
                        cutoff = int(current_frame_idx) + int(keep_frames)
                        to_drop = [
                            t
                            for t in list(non_cond.keys())
                            if int(t) > cutoff and t not in consolidated_non_cond
                        ]
                    else:
                        cutoff = int(current_frame_idx) - int(keep_frames)
                        to_drop = [
                            t
                            for t in list(non_cond.keys())
                            if int(t) < cutoff and t not in consolidated_non_cond
                        ]
                    if not to_drop:
                        return
                    for t in to_drop:
                        non_cond.pop(t, None)
                    output_dict_per_obj = state.get("output_dict_per_obj", None)
                    if isinstance(output_dict_per_obj, dict):
                        for obj_out in output_dict_per_obj.values():
                            d = obj_out.get("non_cond_frame_outputs", None) if isinstance(obj_out, dict) else None
                            if isinstance(d, dict):
                                for t in to_drop:
                                    d.pop(t, None)
                    frames_already_tracked = state.get("frames_already_tracked", None)
                    if isinstance(frames_already_tracked, dict):
                        for t in to_drop:
                            frames_already_tracked.pop(t, None)

                gen = orig_propagate_in_video(
                    self,
                    inference_state,
                    start_frame_idx,
                    max_frame_num_to_track,
                    reverse,
                    tqdm_disable=tqdm_disable,
                    obj_ids=obj_ids,
                    run_mem_encoder=run_mem_encoder,
                    propagate_preflight=propagate_preflight,
                )
                for out in gen:
                    yield out
                    try:
                        _prune(inference_state, int(out[0]), bool(reverse))
                    except Exception:
                        pass

            tracker_cls.propagate_in_video = propagate_in_video_patched
            tracker_cls._simpai_prune_noncond_patched = True
    except Exception:
        pass
    model._simpai_webui_patched = True


def _sync_bias_dtype_with_weight(model) -> None:
    try:
        with torch.no_grad():
            for module in model.modules():
                weight = getattr(module, "weight", None)
                bias = getattr(module, "bias", None)
                if weight is None or bias is None:
                    continue
                if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
                    continue
                if not weight.is_floating_point() or not bias.is_floating_point():
                    continue
                if bias.dtype == weight.dtype:
                    continue
                try:
                    bias.data = bias.data.to(device=weight.device, dtype=weight.dtype)
                except Exception:
                    pass
    except Exception:
        pass


def _choose_video_loader_type(prefer_torchcodec: bool = True) -> str:
    if prefer_torchcodec and importlib.util.find_spec("torchcodec") is not None:
        return "torchcodec"
    return "cv2"


def _precision_to_dtype(precision: str) -> torch.dtype:
    p = str(precision or "").strip().lower()
    if p in ("fp16", "float16", "half", "16"):
        return torch.float16
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _autocast_context(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _apply_video_model_defaults(
    model,
    *,
    score_threshold_detection: float,
    new_det_thresh: float,
    assoc_iou_thresh: float,
    det_nms_thresh: float,
    hotstart_delay: int,
    hotstart_unmatch_thresh: int,
    hotstart_dup_thresh: int,
    suppress_unmatched_only_within_hotstart: bool,
    min_trk_keep_alive: int,
    max_trk_keep_alive: int,
    init_trk_keep_alive: int,
    suppress_overlapping_based_on_recent_occlusion_threshold: float,
    suppress_det_close_to_boundary: bool,
    fill_hole_area: int,
    recondition_every_nth_frame: int,
    masklet_confirmation_enable: bool,
    decrease_trk_keep_alive_for_empty_masklets: bool,
    image_size: int,
) -> None:
    if model is None:
        return
    for k, v in {
        "score_threshold_detection": float(score_threshold_detection),
        "new_det_thresh": float(new_det_thresh),
        "assoc_iou_thresh": float(assoc_iou_thresh),
        "det_nms_thresh": float(det_nms_thresh),
        "hotstart_delay": int(hotstart_delay),
        "hotstart_unmatch_thresh": int(hotstart_unmatch_thresh),
        "hotstart_dup_thresh": int(hotstart_dup_thresh),
        "suppress_unmatched_only_within_hotstart": bool(suppress_unmatched_only_within_hotstart),
        "min_trk_keep_alive": int(min_trk_keep_alive),
        "max_trk_keep_alive": int(max_trk_keep_alive),
        "init_trk_keep_alive": int(init_trk_keep_alive),
        "suppress_overlapping_based_on_recent_occlusion_threshold": float(
            suppress_overlapping_based_on_recent_occlusion_threshold
        ),
        "suppress_det_close_to_boundary": bool(suppress_det_close_to_boundary),
        "fill_hole_area": int(fill_hole_area),
        "recondition_every_nth_frame": int(recondition_every_nth_frame),
        "masklet_confirmation_enable": bool(masklet_confirmation_enable),
        "decrease_trk_keep_alive_for_empty_masklets": bool(
            decrease_trk_keep_alive_for_empty_masklets
        ),
        "image_size": int(image_size),
    }.items():
        if hasattr(model, k):
            try:
                setattr(model, k, v)
            except Exception:
                pass


def _postprocess_mask_u8(mask_u8: np.ndarray, *, strength: int, min_area: int) -> np.ndarray:
    if mask_u8 is None:
        return mask_u8
    s = int(strength or 0)
    if s <= 0:
        return mask_u8
    m = (mask_u8 > 127).astype(np.uint8) * 255

    s = max(1, min(5, s))
    close_ks = 2 * s + 1
    open_ks = max(0, 2 * (s - 2) + 1) if s >= 3 else 0

    if close_ks >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if open_ks >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    area = int(min_area or 0)
    if area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (m > 0).astype(np.uint8), connectivity=8
        )
        keep = np.zeros_like(m, dtype=np.uint8)
        for i in range(1, int(num)):
            if int(stats[i, cv2.CC_STAT_AREA]) >= area:
                keep[labels == i] = 255
        m = keep

    return (m > 127).astype(np.uint8) * 255


def _safe_get_video_file_size(video_path: str) -> int | None:
    try:
        return int(os.path.getsize(video_path))
    except Exception:
        return None


def _collect_model_params(model) -> dict:
    keys = [
        "score_threshold_detection",
        "new_det_thresh",
        "assoc_iou_thresh",
        "det_nms_thresh",
        "hotstart_delay",
        "hotstart_unmatch_thresh",
        "hotstart_dup_thresh",
        "suppress_unmatched_only_within_hotstart",
        "min_trk_keep_alive",
        "max_trk_keep_alive",
        "init_trk_keep_alive",
        "suppress_overlapping_based_on_recent_occlusion_threshold",
        "suppress_det_close_to_boundary",
        "fill_hole_area",
        "recondition_every_nth_frame",
        "masklet_confirmation_enable",
        "decrease_trk_keep_alive_for_empty_masklets",
        "image_size",
    ]
    out = {}
    if model is None:
        return out
    for k in keys:
        if hasattr(model, k):
            try:
                out[k] = getattr(model, k)
            except Exception:
                pass
    return out


def _open_mask_video_writer(out_path: str, *, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ext = os.path.splitext(str(out_path))[1].lower()
    if ext == ".avi":
        candidates = ["MJPG", "XVID", "I420"]
    else:
        candidates = ["avc1", "H264", "X264", "mp4v"]
    for fourcc_text in candidates:
        try:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*fourcc_text),
                float(fps),
                (int(width), int(height)),
            )
            if writer.isOpened():
                return writer
        except Exception:
            pass
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    return writer


def _get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    from shutil import which

    return which("ffmpeg")


def _remux_or_reencode_for_browser_playback(src_path: str, dst_path: str, *, fps: float) -> bool:
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        return False
    try:
        import subprocess

        gop = max(1, int(round(float(fps or 30.0))))
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "1",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except Exception:
        return False


def run_sam3_video_mask(
    video_path: str,
    editor_payload_json: str,
    propagation_direction: str = "both",
    max_frames_to_track: int = -1,
    close_after_propagation: bool = True,
    precision: str = "fp16",
    score_threshold_detection: float = 0.3,
    new_det_thresh: float = 0.7,
    det_nms_thresh: float = 0.1,
    fill_hole_area: int = 16,
    recondition_every_nth_frame: int = 16,
    image_size: int = 1008,
    postprocess_strength: int = 0,
    postprocess_min_area: int = 0,
    debug_print: bool = False,
    force_browser_compatible_output: bool = True,
    invert_mask: bool = False,
    unload_after_run: bool = True,
) -> str:
    if video_path is None or not str(video_path).strip():
        raise ValueError("video_path is empty")
    video_path = os.path.abspath(video_path)

    predictor = None
    model = None
    session_id = None
    tmp_frames_dir = None
    resource_path = video_path
    masks_by_frame: dict[int, np.ndarray] = {}
    try:
        checkpoint_path = _resolve_sam3_checkpoint(None)
        predictor = _load_video_predictor(checkpoint_path)
        model = getattr(predictor, "model", None)
        dtype = _precision_to_dtype(precision)
        _apply_video_model_defaults(
            model,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            assoc_iou_thresh=0.1,
            det_nms_thresh=det_nms_thresh,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=fill_hole_area,
            recondition_every_nth_frame=recondition_every_nth_frame,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=image_size,
        )
        _sync_bias_dtype_with_weight(model)

        fps, frame_count, width, height = _get_video_meta(video_path)
        payload = _parse_editor_payload(editor_payload_json)
        offload_video_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_VIDEO_TO_CPU", True)
        async_loading_frames = _env_flag("SIMPLEAI_SAM3_ASYNC_LOADING_FRAMES", True)
        offload_state_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_STATE_TO_CPU", True)
        offload_cached_masks_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_CACHED_MASKS_TO_CPU", True)
        if debug_print:
            logger.info(
                "SAM3(webui) video=%s size_bytes=%s fps=%s frames=%s w=%s h=%s precision=%s dtype=%s model_params=%s offload_video_to_cpu=%s async_loading_frames=%s offload_state_to_cpu=%s offload_cached_masks_to_cpu=%s elastic_vram=%s",
                str(video_path),
                str(_safe_get_video_file_size(video_path)),
                float(fps),
                int(frame_count),
                int(width),
                int(height),
                str(precision),
                str(dtype),
                str(_collect_model_params(model)),
                bool(offload_video_to_cpu),
                bool(async_loading_frames),
                bool(offload_state_to_cpu),
                bool(offload_cached_masks_to_cpu),
                str(_apply_elastic_vram_limit_for_sam3(device_index=int(torch.cuda.current_device()))),
            )

        frame_time = float(payload.get("frame_time", 0.0) or 0.0)
        frame_index = int(round(frame_time * fps)) if fps > 0 else int(payload.get("frame_index", 0) or 0)
        frame_index = max(0, min(frame_count - 1, frame_index)) if frame_count > 0 else max(0, frame_index)

        pos_points = _normalize_points(payload.get("positive_coords", []), width, height)
        neg_points = _normalize_points(payload.get("negative_coords", []), width, height)
        bbox_xywh = _normalize_bbox(payload.get("bbox", None), width, height)
        points, point_labels = [], []
        if pos_points: points.extend(pos_points); point_labels.extend([1] * len(pos_points))
        if neg_points: points.extend(neg_points); point_labels.extend([0] * len(neg_points))

        prompt_text = str(payload.get("prompt", "") or "").strip()
        if not bbox_xywh and not points and not prompt_text:
            raise ValueError("No prompt provided. Please add points in editor or provide prompt text.")

        preferred_obj_id = 1 if (points and not bbox_xywh) else None

        with torch.inference_mode(), _autocast_context(dtype):
            try:
                predictor.async_loading_frames = bool(async_loading_frames)
                predictor.video_loader_type = _choose_video_loader_type()
            except Exception:
                pass
            if (
                getattr(predictor, "video_loader_type", "cv2") != "torchcodec"
                and getattr(predictor, "async_loading_frames", False)
                and _is_video_file(video_path)
                and os.path.isfile(video_path)
            ):
                tmp_frames_dir = _extract_video_to_temp_frames(
                    video_path, max_frames=int(frame_count) if int(frame_count) > 0 else -1
                )
                resource_path = tmp_frames_dir
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resource_path,
                    session_id=None,
                    offload_video_to_cpu=bool(offload_video_to_cpu),
                    async_loading_frames=bool(async_loading_frames),
                    video_loader_type=_choose_video_loader_type(),
                )
            )
            session_id = response.get("session_id", None)
            if not session_id:
                raise RuntimeError("Failed to start SAM3 session")

            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=None if (bbox_xywh or points) else (prompt_text or None),
                    points=points if (points and not bbox_xywh) else None,
                    point_labels=point_labels if (points and not bbox_xywh) else None,
                    bounding_boxes=bbox_xywh if bbox_xywh else None,
                    bounding_box_labels=([1] * len(bbox_xywh)) if bbox_xywh else None,
                    obj_id=1,
                )
            )

            for r in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=propagation_direction,
                    start_frame_index=frame_index,
                    max_frame_num_to_track=None if max_frames_to_track == -1 else int(max_frames_to_track),
                )
            ):
                frame_idx = int(r.get("frame_index", 0) or 0)
                outputs = r.get("outputs", {}) or {}
                merged = _select_binary_mask_from_outputs(outputs, preferred_obj_id=preferred_obj_id)
                if merged is None:
                    continue
                m = _postprocess_mask_u8(merged.astype(np.uint8) * 255, strength=postprocess_strength, min_area=postprocess_min_area)
                masks_by_frame[frame_idx] = m

            if debug_print:
                try:
                    if masks_by_frame:
                        keys = sorted(masks_by_frame.keys())
                        expected_total = frame_count if frame_count > 0 else (int(keys[-1]) + 1)
                        first_missing = [i for i in range(int(min(30, expected_total))) if i not in masks_by_frame]
                        logger.info(
                            "SAM3(webui) masks: frames_with_mask=%s min=%s max=%s expected_total=%s missing_first=%s",
                            int(len(keys)),
                            int(keys[0]),
                            int(keys[-1]),
                            int(expected_total),
                            str(first_missing),
                        )
                    else:
                        logger.info("SAM3(webui) masks: empty (no frames received)")
                except Exception:
                    logger.exception("SAM3(webui) masks: failed to summarize frame coverage")
    finally:
        if close_after_propagation and predictor is not None and session_id:
            try:
                predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            except Exception:
                pass
        if tmp_frames_dir:
            try:
                import shutil

                shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            del model
        except Exception:
            pass
        try:
            del predictor
        except Exception:
            pass

    result = None
    try:
        out_dir = os.path.join(_repo_root(), "outputs", "sam3_masks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sam3_mask_{uuid.uuid4().hex}.mp4")
        ffmpeg_ok = bool(_get_ffmpeg_exe()) if force_browser_compatible_output else False
        raw_ext = ".avi" if ffmpeg_ok else ".mp4"
        raw_out_path = os.path.join(out_dir, f"sam3_mask_{uuid.uuid4().hex}_raw{raw_ext}")

        if width <= 0 or height <= 0:
            if masks_by_frame:
                any_mask = next(iter(masks_by_frame.values()))
                height, width = int(any_mask.shape[0]), int(any_mask.shape[1])
            else:
                raise RuntimeError("Could not infer video dimensions for output.")

        out_fps = fps if fps > 0 else 24.0
        writer = _open_mask_video_writer(raw_out_path, fps=float(out_fps), width=int(width), height=int(height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for mask output.")

        total = frame_count if frame_count > 0 else (max(masks_by_frame.keys()) + 1 if masks_by_frame else 0)
        for i in range(int(total)):
            mask = masks_by_frame.get(i, None)
            if mask is None:
                frame = np.zeros((height, width), dtype=np.uint8)
            else:
                if mask.shape[0] != height or mask.shape[1] != width:
                    frame = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    frame = mask
            if invert_mask:
                frame = 255 - frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
        writer.release()

        if force_browser_compatible_output:
            ok = _remux_or_reencode_for_browser_playback(raw_out_path, out_path, fps=float(out_fps))
            if ok:
                try:
                    if os.path.isfile(raw_out_path):
                        os.remove(raw_out_path)
                except Exception:
                    pass
                if debug_print:
                    logger.info("SAM3(webui) mask video output: %s", str(out_path))
                result = out_path
            else:
                try:
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                except Exception:
                    pass
                if debug_print:
                    logger.info("SAM3(webui) mask video output: %s", str(raw_out_path))
                result = raw_out_path
        else:
            result = out_path
        return result
    finally:
        try:
            del masks_by_frame
        except Exception:
            pass
        if unload_after_run:
            unload_sam3_video_predictor()


def run_sam3_video_mask_by_prompt(
    video_path: str,
    prompt: str,
    frame_time: float = 0.0,
    propagation_direction: str = "both",
    max_frames_to_track: int = -1,
    close_after_propagation: bool = True,
    precision: str = "fp16",
    score_threshold_detection: float = 0.3,
    new_det_thresh: float = 0.7,
    det_nms_thresh: float = 0.1,
    fill_hole_area: int = 16,
    recondition_every_nth_frame: int = 16,
    image_size: int = 1008,
    postprocess_strength: int = 0,
    postprocess_min_area: int = 0,
    debug_print: bool = False,
    force_browser_compatible_output: bool = True,
    invert_mask: bool = False,
    unload_after_run: bool = True,
) -> str:
    if video_path is None or not str(video_path).strip():
        raise ValueError("video_path is empty")
    video_path = os.path.abspath(video_path)

    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")

    predictor = None
    model = None
    session_id = None
    tmp_frames_dir = None
    resource_path = video_path
    masks_by_frame: dict[int, np.ndarray] = {}
    try:
        checkpoint_path = _resolve_sam3_checkpoint(None)
        predictor = _load_video_predictor(checkpoint_path)
        model = getattr(predictor, "model", None)
        dtype = _precision_to_dtype(precision)
        _apply_video_model_defaults(
            model,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            assoc_iou_thresh=0.1,
            det_nms_thresh=det_nms_thresh,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=fill_hole_area,
            recondition_every_nth_frame=recondition_every_nth_frame,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=image_size,
        )
        _sync_bias_dtype_with_weight(model)

        fps, frame_count, width, height = _get_video_meta(video_path)
        offload_video_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_VIDEO_TO_CPU", True)
        async_loading_frames = _env_flag("SIMPLEAI_SAM3_ASYNC_LOADING_FRAMES", True)
        offload_state_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_STATE_TO_CPU", True)
        offload_cached_masks_to_cpu = _env_flag("SIMPLEAI_SAM3_OFFLOAD_CACHED_MASKS_TO_CPU", True)
        frame_index = int(round(float(frame_time or 0.0) * fps)) if fps > 0 else 0
        frame_index = max(0, min(frame_count - 1, frame_index)) if frame_count > 0 else max(0, frame_index)
        if debug_print:
            logger.info(
                "SAM3(webui) video=%s size_bytes=%s fps=%s frames=%s w=%s h=%s precision=%s dtype=%s model_params=%s offload_video_to_cpu=%s async_loading_frames=%s offload_state_to_cpu=%s offload_cached_masks_to_cpu=%s elastic_vram=%s prompt=%s",
                str(video_path),
                str(_safe_get_video_file_size(video_path)),
                float(fps),
                int(frame_count),
                int(width),
                int(height),
                str(precision),
                str(dtype),
                str(_collect_model_params(model)),
                bool(offload_video_to_cpu),
                bool(async_loading_frames),
                bool(offload_state_to_cpu),
                bool(offload_cached_masks_to_cpu),
                str(_apply_elastic_vram_limit_for_sam3(device_index=int(torch.cuda.current_device()))),
                str(prompt),
            )

        with torch.inference_mode(), _autocast_context(dtype):
            try:
                predictor.async_loading_frames = bool(async_loading_frames)
                predictor.video_loader_type = _choose_video_loader_type()
            except Exception:
                pass
            if (
                getattr(predictor, "video_loader_type", "cv2") != "torchcodec"
                and getattr(predictor, "async_loading_frames", False)
                and _is_video_file(video_path)
                and os.path.isfile(video_path)
            ):
                tmp_frames_dir = _extract_video_to_temp_frames(
                    video_path, max_frames=int(frame_count) if int(frame_count) > 0 else -1
                )
                resource_path = tmp_frames_dir
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resource_path,
                    session_id=None,
                    offload_video_to_cpu=bool(offload_video_to_cpu),
                    async_loading_frames=bool(async_loading_frames),
                    video_loader_type=_choose_video_loader_type(),
                )
            )
            session_id = response.get("session_id", None)
            if not session_id:
                raise RuntimeError("Failed to start SAM3 session")

            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=prompt,
                    points=None,
                    point_labels=None,
                    bounding_boxes=None,
                    bounding_box_labels=None,
                    obj_id=1,
                )
            )

            for r in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=propagation_direction,
                    start_frame_index=frame_index,
                    max_frame_num_to_track=None if max_frames_to_track == -1 else int(max_frames_to_track),
                )
            ):
                frame_idx = int(r.get("frame_index", 0) or 0)
                outputs = r.get("outputs", {}) or {}
                mask = outputs.get("out_binary_masks", None)
                if mask is None:
                    continue
                if isinstance(mask, np.ndarray) and mask.ndim == 3 and mask.shape[0] > 0:
                    merged = np.any(mask, axis=0)
                elif isinstance(mask, np.ndarray) and mask.ndim == 2:
                    merged = mask.astype(bool)
                else:
                    continue
                m = _postprocess_mask_u8(merged.astype(np.uint8) * 255, strength=postprocess_strength, min_area=postprocess_min_area)
                masks_by_frame[frame_idx] = m

            if debug_print:
                try:
                    if masks_by_frame:
                        keys = sorted(masks_by_frame.keys())
                        expected_total = frame_count if frame_count > 0 else (int(keys[-1]) + 1)
                        first_missing = [i for i in range(int(min(30, expected_total))) if i not in masks_by_frame]
                        logger.info(
                            "SAM3(webui) masks: frames_with_mask=%s min=%s max=%s expected_total=%s missing_first=%s",
                            int(len(keys)),
                            int(keys[0]),
                            int(keys[-1]),
                            int(expected_total),
                            str(first_missing),
                        )
                    else:
                        logger.info("SAM3(webui) masks: empty (no frames received)")
                except Exception:
                    logger.exception("SAM3(webui) masks: failed to summarize frame coverage")
    finally:
        if close_after_propagation and predictor is not None and session_id:
            try:
                predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            except Exception:
                pass
        if tmp_frames_dir:
            try:
                import shutil

                shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            del model
        except Exception:
            pass
        try:
            del predictor
        except Exception:
            pass

    result = None
    try:
        out_dir = os.path.join(_repo_root(), "outputs", "sam3_masks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sam3_mask_{uuid.uuid4().hex}.mp4")
        ffmpeg_ok = bool(_get_ffmpeg_exe()) if force_browser_compatible_output else False
        raw_ext = ".avi" if ffmpeg_ok else ".mp4"
        raw_out_path = os.path.join(out_dir, f"sam3_mask_{uuid.uuid4().hex}_raw{raw_ext}")

        if width <= 0 or height <= 0:
            if masks_by_frame:
                any_mask = next(iter(masks_by_frame.values()))
                height, width = int(any_mask.shape[0]), int(any_mask.shape[1])
            else:
                raise RuntimeError("Could not infer video dimensions for output.")

        out_fps = fps if fps > 0 else 24.0
        writer = _open_mask_video_writer(raw_out_path, fps=float(out_fps), width=int(width), height=int(height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for mask output.")

        total = frame_count if frame_count > 0 else (max(masks_by_frame.keys()) + 1 if masks_by_frame else 0)
        for i in range(int(total)):
            mask = masks_by_frame.get(i, None)
            if mask is None:
                frame = np.zeros((height, width), dtype=np.uint8)
            else:
                if mask.shape[0] != height or mask.shape[1] != width:
                    frame = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    frame = mask
            if invert_mask:
                frame = 255 - frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
        writer.release()

        if force_browser_compatible_output:
            ok = _remux_or_reencode_for_browser_playback(raw_out_path, out_path, fps=float(out_fps))
            if ok:
                try:
                    if os.path.isfile(raw_out_path):
                        os.remove(raw_out_path)
                except Exception:
                    pass
                if debug_print:
                    logger.info("SAM3(webui) mask video output: %s", str(out_path))
                result = out_path
            else:
                try:
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                except Exception:
                    pass
                if debug_print:
                    logger.info("SAM3(webui) mask video output: %s", str(raw_out_path))
                result = raw_out_path
        else:
            result = out_path
        return result
    finally:
        try:
            del masks_by_frame
        except Exception:
            pass
        if unload_after_run:
            unload_sam3_video_predictor()


def get_viewer_html() -> str:
    glue_js = r"""
<img src="x" style="display:none" onerror='(function(){
  if (window.sam3FramesEditorInitialized) return;
  window.sam3FramesEditorInitialized = true;

  const state = {
    open: false,
    mode: "point",
    pointsPos: [],
    pointsNeg: [],
    box: null,
    boxDraft: null,
    history: [],
    historyIndex: -1,
    duration: 0,
    videoWidth: 0,
    videoHeight: 0,
    time: 0,
  };

  const $ = (id) => document.getElementById(id);
  const backdrop = $("sam3_frames_modal_backdrop");
  const canvas = $("sam3_canvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  const hiddenVideo = $("sam3_hidden_video");
  const slider = $("sam3_time_slider");
  const label = $("sam3_frame_label");
  const btnUndo = $("sam3_undo");
  const btnRedo = $("sam3_redo");
  const btnClear = $("sam3_clear");
  const btnModePoint = $("sam3_mode_point");
  const btnModeBox = $("sam3_mode_box");
  const btnCancel = $("sam3_cancel");
  const btnConfirm = $("sam3_confirm");
  const body = $("sam3_frames_body");

  if (!backdrop || !canvas || !ctx || !hiddenVideo || !slider || !label || !btnUndo || !btnRedo || !btnClear || !btnModePoint || !btnModeBox || !btnCancel || !btnConfirm || !body) {
    return;
  }

  const layoutCanvas = () => {
    if (!state.open) return;
    if (state.videoWidth <= 0 || state.videoHeight <= 0) return;
    const bw = Math.max(1, body.clientWidth || 1);
    const bh = Math.max(1, body.clientHeight || 1);
    const aspect = state.videoWidth / state.videoHeight;
    let w = bw;
    let h = Math.round(w / aspect);
    if (h > bh) {
      h = bh;
      w = Math.round(h * aspect);
    }
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  };

  const updateUndoRedo = () => {
    btnUndo.style.opacity = state.historyIndex >= 0 ? "1" : "0.35";
    btnUndo.style.pointerEvents = state.historyIndex >= 0 ? "auto" : "none";
    btnRedo.style.opacity = state.historyIndex < state.history.length - 1 ? "1" : "0.35";
    btnRedo.style.pointerEvents = state.historyIndex < state.history.length - 1 ? "auto" : "none";
  };

  const draw = () => {
    if (!state.open) return;
    if (state.videoWidth <= 0 || state.videoHeight <= 0) return;
    canvas.width = state.videoWidth;
    canvas.height = state.videoHeight;
    ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);

    const drawBox = (b, color) => {
      const x0 = b.x0 * canvas.width;
      const y0 = b.y0 * canvas.height;
      const x1 = b.x1 * canvas.width;
      const y1 = b.y1 * canvas.height;
      const x = Math.min(x0, x1);
      const y = Math.min(y0, y1);
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      if (w <= 0 || h <= 0) return;
      ctx.lineWidth = Math.max(2, canvas.width * 0.003);
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(45, 212, 191, 0.12)";
      ctx.fillRect(x, y, w, h);
    };

    const drawPoint = (p, color) => {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, Math.max(6, canvas.width * 0.008), 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.stroke();
    };
    state.pointsPos.forEach(p => drawPoint(p, "#70FF81"));
    state.pointsNeg.forEach(p => drawPoint(p, "#FF6B6B"));
    if (state.box) drawBox(state.box, "rgba(45, 212, 191, 1)");
    if (state.boxDraft) drawBox(state.boxDraft, "rgba(45, 212, 191, 0.9)");
  };

  const seekToTime = (t, skipSliderUpdate) => {
    if (!state.open) return;
    const clamped = Math.max(0, Math.min(state.duration || 0, t));
    state.time = clamped;
    if (!skipSliderUpdate && state.duration > 0) {
      slider.value = String(Math.round((clamped / state.duration) * 1000));
    }
    label.textContent = "t=" + clamped.toFixed(2) + "s";
    hiddenVideo.currentTime = clamped;
  };

  const pushHistory = () => {
    const snapshot = {
      mode: state.mode,
      pointsPos: state.pointsPos.map(p => ({x:p.x,y:p.y})),
      pointsNeg: state.pointsNeg.map(p => ({x:p.x,y:p.y})),
      box: state.box ? {x0:state.box.x0,y0:state.box.y0,x1:state.box.x1,y1:state.box.y1} : null,
      time: state.time,
    };
    if (state.historyIndex < state.history.length - 1) {
      state.history = state.history.slice(0, state.historyIndex + 1);
    }
    state.history.push(snapshot);
    state.historyIndex = state.history.length - 1;
    updateUndoRedo();
  };

  const restoreHistory = (idx) => {
    const s = state.history[idx];
    state.mode = s.mode || "point";
    state.pointsPos = (s.pointsPos || []).map(p => ({x:p.x,y:p.y}));
    state.pointsNeg = (s.pointsNeg || []).map(p => ({x:p.x,y:p.y}));
    state.box = s.box ? {x0:s.box.x0,y0:s.box.y0,x1:s.box.x1,y1:s.box.y1} : null;
    state.boxDraft = null;
    state.time = s.time || 0;
    seekToTime(state.time, true);
    updateUndoRedo();
    btnModePoint.classList.toggle("active", state.mode === "point");
    btnModeBox.classList.toggle("active", state.mode === "box");
  };

  slider.addEventListener("input", () => {
    if (!state.open) return;
    if (state.duration <= 0) return;
    const v = Number(slider.value || 0);
    const t = (v / 1000) * state.duration;
    seekToTime(t, true);
  });

  hiddenVideo.addEventListener("seeked", () => draw());
  hiddenVideo.addEventListener("loadedmetadata", () => {
    state.duration = hiddenVideo.duration || 0;
    state.videoWidth = hiddenVideo.videoWidth || 0;
    state.videoHeight = hiddenVideo.videoHeight || 0;
    layoutCanvas();
    seekToTime(0, false);
    draw();
  });

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  const getCanvasNormPos = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  };

  let dragActive = false;

  canvas.addEventListener("mousedown", (e) => {
    if (!state.open) return;
    if (state.videoWidth <= 0 || state.videoHeight <= 0) return;
    const p = getCanvasNormPos(e);

    if (state.mode === "box") {
      if (e.button === 2) {
        state.box = null;
        state.boxDraft = null;
        pushHistory();
        draw();
        return;
      }
      if (e.button !== 0) return;
      dragActive = true;
      state.boxDraft = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      draw();
      return;
    }

    if (e.button === 2) state.pointsNeg.push(p);
    else state.pointsPos.push(p);
    pushHistory();
    draw();
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.open) return;
    if (!dragActive) return;
    if (state.mode !== "box") return;
    if (!state.boxDraft) return;
    const p = getCanvasNormPos(e);
    state.boxDraft.x1 = p.x;
    state.boxDraft.y1 = p.y;
    draw();
  });

  window.addEventListener("mouseup", (e) => {
    if (!state.open) return;
    if (!dragActive) return;
    dragActive = false;
    if (state.mode !== "box") return;
    if (!state.boxDraft) return;
    const b = state.boxDraft;
    state.boxDraft = null;
    const w = Math.abs(b.x1 - b.x0);
    const h = Math.abs(b.y1 - b.y0);
    if (w > 0.002 && h > 0.002) {
      state.box = b;
      pushHistory();
    }
    draw();
  });

  btnUndo.addEventListener("click", () => {
    if (state.historyIndex >= 0) {
      state.historyIndex -= 1;
      if (state.historyIndex >= 0) restoreHistory(state.historyIndex);
      else {
        state.pointsPos = [];
        state.pointsNeg = [];
        updateUndoRedo();
        draw();
      }
    }
  });
  btnRedo.addEventListener("click", () => {
    if (state.historyIndex < state.history.length - 1) {
      state.historyIndex += 1;
      restoreHistory(state.historyIndex);
    }
  });
  btnClear.addEventListener("click", () => {
    state.pointsPos = [];
    state.pointsNeg = [];
    state.box = null;
    state.boxDraft = null;
    pushHistory();
    draw();
  });

  btnModePoint.addEventListener("click", () => {
    state.mode = "point";
    btnModePoint.classList.add("active");
    btnModeBox.classList.remove("active");
    pushHistory();
    draw();
  });

  btnModeBox.addEventListener("click", () => {
    state.mode = "box";
    btnModeBox.classList.add("active");
    btnModePoint.classList.remove("active");
    pushHistory();
    draw();
  });

  const closeModal = () => {
    state.open = false;
    backdrop.style.display = "none";
  };

  btnCancel.addEventListener("click", () => closeModal());
  backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) closeModal(); });

  btnConfirm.addEventListener("click", () => {
    const payload = {
      frame_time: state.time,
      positive_coords: state.pointsPos.map(p => ({x:p.x, y:p.y})),
      negative_coords: state.pointsNeg.map(p => ({x:p.x, y:p.y})),
      bbox: state.box ? {x0:state.box.x0,y0:state.box.y0,x1:state.box.x1,y1:state.box.y1} : null,
    };
    const hidden = document.querySelector("#sam3_editor_payload textarea, #sam3_editor_payload input");
    if (hidden) {
      hidden.value = JSON.stringify(payload);
      hidden.dispatchEvent(new Event("input", { bubbles: true }));
      hidden.dispatchEvent(new Event("change", { bubbles: true }));
    }
    const genBtn = document.querySelector("#sam3_points_generate_btn button, #sam3_points_generate_btn");
    if (genBtn) genBtn.click();
    closeModal();
  });

  const getVideoSrc = (videoEl) => {
    if (!videoEl) return "";
    const src = videoEl.currentSrc || videoEl.src || "";
    if (src) return src;
    const source = videoEl.querySelector("source");
    return (source && source.src) ? source.src : "";
  };

  const openFromVideo = (videoEl) => {
    const src = getVideoSrc(videoEl);
    if (!src) return;
    state.mode = "point";
    state.pointsPos = [];
    state.pointsNeg = [];
    state.box = null;
    state.boxDraft = null;
    state.history = [];
    state.historyIndex = -1;
    updateUndoRedo();
    btnModePoint.classList.toggle("active", state.mode === "point");
    btnModeBox.classList.toggle("active", state.mode === "box");
    hiddenVideo.src = src;
    hiddenVideo.load();
    state.open = true;
    backdrop.style.display = "flex";
    layoutCanvas();
  };

  window.addEventListener("resize", () => layoutCanvas());
  try {
    const ro = new ResizeObserver(() => layoutCanvas());
    ro.observe(body);
  } catch (e) {}

  const bindClick = () => {
    const container = document.getElementById("sam3_input_video");
    if (!container) return false;
    const v = container.querySelector("video");
    if (!v) return false;
    if (v.dataset.sam3Bound === "1") return true;
    v.dataset.sam3Bound = "1";

    const handler = (e) => {
      if (e.defaultPrevented) return;
      if (e.button !== 0) return;
      if (e.target !== v) return;
      openFromVideo(v);
    };

    v.addEventListener("dblclick", handler, false);
    return true;
  };

  const mo = new MutationObserver(() => bindClick());
  mo.observe(document.documentElement, { childList: true, subtree: true });
  bindClick();
})()'>
"""

    return r"""
<div id="sam3_frames_editor_root"></div>
<style>
  #sam3_frames_modal_backdrop{
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    display: none;
    align-items: flex-start;
    justify-content: center;
    overflow: auto;
    padding: 24px 0;
    z-index: 99999;
  }
  #sam3_frames_modal{
    width: min(980px, 92vw);
    height: min(720px, calc(100vh - 48px));
    background: #0f1011;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  #sam3_frames_toolbar{
    flex: 0 0 36px;
    width: 100%;
    background: #222;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 6px;
    box-sizing: border-box;
    border-bottom: 1px solid #333;
  }
  #sam3_frames_hint{
    flex: 1;
    min-width: 0;
    text-align: center;
    color: #bbb;
    font-size: 12px;
    user-select: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 0 8px;
  }
  .sam3_hint_dot{
    display: inline-block;
    width: 10px;
    text-align: center;
    font-weight: 700;
  }
  .sam3_btn{
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-radius: 6px;
    color: #ccc;
    background: transparent;
    border: 1px solid transparent;
    user-select: none;
  }
  .sam3_btn:hover{ background:#333; }
  .sam3_btn.active{ background:#444; color:#fff; border-color:#555; }
  #sam3_frames_body{
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f1011;
    min-height: 0;
    overflow: auto;
  }
  #sam3_canvas{
    max-width: 100%;
    max-height: 100%;
    cursor: crosshair;
    display: block;
  }
  #sam3_frames_footer{
    flex: 0 0 44px;
    width: 100%;
    background: #222;
    border-top: 1px solid #333;
    padding: 6px 10px;
    box-sizing: border-box;
    display: flex;
    gap: 10px;
    align-items: center;
  }
  #sam3_time_slider{ flex: 1; height: 4px; }
  #sam3_frame_label{ color:#ccc; font-family: monospace; font-size: 12px; min-width: 140px; text-align:center; user-select:none; }
  #sam3_confirm{
    height: 30px;
    padding: 0 12px;
    border-radius: 6px;
    border: 1px solid #555;
    background: #2b6cb0;
    color: #fff;
    cursor: pointer;
  }
  #sam3_cancel{
    height: 30px;
    padding: 0 12px;
    border-radius: 6px;
    border: 1px solid #555;
    background: #333;
    color: #eee;
    cursor: pointer;
  }
</style>

<div id="sam3_frames_modal_backdrop">
  <div id="sam3_frames_modal">
    <div id="sam3_frames_toolbar">
      <div style="display:flex; gap:6px; align-items:center;">
        <div class="sam3_btn" id="sam3_undo" title="Undo">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/></svg>
        </div>
        <div class="sam3_btn" id="sam3_redo" title="Redo">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M18.4 10.6C16.55 9 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/></svg>
        </div>
        <div class="sam3_btn" id="sam3_clear" title="Clear All">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
        </div>
      </div>
      <div id="sam3_frames_hint">
        左键<span class="sam3_hint_dot" style="color:#22c55e;">●</span>选取，右键<span class="sam3_hint_dot" style="color:#ef4444;">●</span>排除
      </div>
      <div style="display:flex; gap:6px; align-items:center;">
        <div class="sam3_btn active" id="sam3_mode_point" title="Point Mode">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
        </div>
        <div class="sam3_btn" id="sam3_mode_box" title="Box Mode">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 3h18v18H3V3zm2 2v14h14V5H5z"/></svg>
        </div>
      </div>
    </div>
    <div id="sam3_frames_body">
      <canvas id="sam3_canvas"></canvas>
      <video id="sam3_hidden_video" style="display:none;"></video>
    </div>
    <div id="sam3_frames_footer">
      <div id="sam3_frame_label">t=0.00s</div>
      <input id="sam3_time_slider" type="range" min="0" max="1000" value="0" step="1" />
      <button id="sam3_cancel">取消</button>
      <button id="sam3_confirm">确认</button>
    </div>
  </div>
</div>
""" + glue_js
