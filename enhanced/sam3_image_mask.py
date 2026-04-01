import base64
import io
import json
import os
import sys
import threading
from typing import Any
from typing import Iterable
from typing import Tuple
from typing import cast
import uuid

import numpy as np
from PIL import Image

SAM3_PT_URL = "https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_PROCESSOR: Any = None
_MODEL_DEVICE: str | None = None


def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def _ensure_easy_sam3_importable() -> None:
    easy_sam3_root = os.path.join(
        _repo_root(), "comfy", "custom_nodes", "ComfyUI-Easy-Sam3"
    )
    easy_sam3_root = os.path.abspath(easy_sam3_root)
    if easy_sam3_root not in sys.path:
        sys.path.insert(0, easy_sam3_root)


def _resolve_sam3_checkpoint(model_path: str | None, *, allow_download: bool) -> str | None:
    import modules.config as config

    if model_path:
        model_path = os.path.abspath(str(model_path))
        if os.path.isfile(model_path):
            return model_path

    models_root = config.get_path_models_root()
    model_dir = os.path.abspath(os.path.join(models_root, "sam3"))
    ckpt_path = os.path.abspath(os.path.join(model_dir, "sam3.pt"))

    if os.path.isfile(ckpt_path):
        return ckpt_path
    if not allow_download:
        return None

    from modules.model_loader import load_file_from_url

    os.makedirs(model_dir, exist_ok=True)
    load_file_from_url(url=SAM3_PT_URL, model_dir=model_dir, file_name="sam3.pt")
    return ckpt_path if os.path.isfile(ckpt_path) else None


def ensure_sam3_image_model_loaded(*, model_path: str | None = None) -> None:
    global _MODEL, _PROCESSOR, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return

        ckpt_path = _resolve_sam3_checkpoint(model_path, allow_download=True)
        if not ckpt_path:
            raise FileNotFoundError("SAM3 checkpoint not found (sam3.pt).")

        _ensure_easy_sam3_importable()
        import torch  # noqa: E402

        from sam3.model_builder import build_sam3_image_model  # noqa: E402
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        processor = Sam3Processor(
            model=model,
            resolution=1008,
            device=device,
            confidence_threshold=0.3,
        )

        _MODEL = model
        _PROCESSOR = processor
        _MODEL_DEVICE = device


def offload_sam3_image_model_to_cpu() -> None:
    global _MODEL, _PROCESSOR, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is None:
            _PROCESSOR = None
            _MODEL_DEVICE = None
            return
        try:
            import torch  # noqa: E402

            try:
                _MODEL.to("cpu")
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        finally:
            _MODEL = None
            _PROCESSOR = None
            _MODEL_DEVICE = None


def _normalize_points(points: Iterable[dict[str, Any]] | None) -> list[list[float]]:
    out: list[list[float]] = []
    if not points:
        return out
    for p in points:
        if not isinstance(p, dict):
            continue
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        out.append([x, y])
    return out


def run_sam3_image_mask_from_points(
    image_rgb: Image.Image,
    *,
    positive_points: Iterable[dict[str, Any]] | None,
    negative_points: Iterable[dict[str, Any]] | None,
    threshold: float = 0.3,
    mask_threshold: float = 0.4,
    model_path: str | None = None,
) -> np.ndarray:
    ensure_sam3_image_model_loaded(model_path=model_path)

    processor = cast(Any, _PROCESSOR)
    processor.set_confidence_threshold(float(threshold))

    pos = _normalize_points(positive_points)
    neg = _normalize_points(negative_points)
    points = pos + neg
    labels = ([1] * len(pos)) + ([0] * len(neg))

    state = processor.set_image(image_rgb)
    if len(points) > 0:
        state = processor.add_point_prompt(points, labels, state)
    else:
        state = processor.set_text_prompt("visual", state)

    masks_logits = state.get("masks_logits", None)
    masks = state.get("masks", None)
    scores = state.get("scores", None)
    if (
        (masks_logits is None or getattr(masks_logits, "numel", lambda: 0)() == 0)
        and (masks is None or getattr(masks, "numel", lambda: 0)() == 0)
    ):
        return np.zeros((image_rgb.height, image_rgb.width), dtype=np.uint8)

    import torch  # noqa: E402

    masks_bin = None
    logits_cpu = None
    if masks_logits is not None and getattr(masks_logits, "numel", lambda: 0)() > 0:
        ml = masks_logits
        if getattr(ml, "ndim", 0) == 4 and ml.shape[1] == 1:
            ml = ml[:, 0]
        if getattr(ml, "ndim", 0) == 2:
            ml = ml.unsqueeze(0)
        ml = ml.detach().to("cpu").float()
        logits_cpu = ml
        masks_bin = ml > float(mask_threshold)
    elif masks is not None and getattr(masks, "numel", lambda: 0)() > 0:
        mb = masks
        if getattr(mb, "ndim", 0) == 4 and mb.shape[1] == 1:
            mb = mb[:, 0]
        if getattr(mb, "ndim", 0) == 2:
            mb = mb.unsqueeze(0)
        mb = mb.detach().to("cpu")
        if mb.dtype != torch.bool:
            mb = mb > 0.5
        masks_bin = mb

    if masks_bin is None or masks_bin.numel() == 0:
        return np.zeros((image_rgb.height, image_rgb.width), dtype=np.uint8)

    w = int(image_rgb.width)
    h = int(image_rgb.height)
    pos = _normalize_points(positive_points)
    neg = _normalize_points(negative_points)

    def _to_px(points01: list[list[float]]) -> tuple[list[int], list[int]]:
        xs: list[int] = []
        ys: list[int] = []
        for x01, y01 in points01:
            x = int(round(float(x01) * max(1, w - 1)))
            y = int(round(float(y01) * max(1, h - 1)))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            xs.append(x)
            ys.append(y)
        return xs, ys

    pos_xs, pos_ys = _to_px(pos)
    neg_xs, neg_ys = _to_px(neg)

    src = logits_cpu if logits_cpu is not None else masks_bin.float()
    pos_hit_thr = float(max(0.05, float(mask_threshold) - 0.15)) if logits_cpu is not None else 0.5
    neg_hit_thr = float(mask_threshold) if logits_cpu is not None else 0.5
    hit_radius = 2

    def _neigh_max(t: Any, x: int, y: int, r: int) -> Any:
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        patch = t[:, y0:y1, x0:x1]
        if patch.numel() == 0:
            return torch.zeros((t.shape[0],), dtype=torch.float32)
        return patch.amax(dim=(1, 2))

    neg_violation = torch.zeros((masks_bin.shape[0],), dtype=torch.bool)
    if len(neg_xs) > 0:
        for x, y in zip(neg_xs, neg_ys):
            neg_violation |= (_neigh_max(src, x, y, hit_radius) > neg_hit_thr)

    selected: set[int] = set()
    if len(pos_xs) > 0:
        for x, y in zip(pos_xs, pos_ys):
            strengths = _neigh_max(src, x, y, hit_radius)
            strengths = strengths.masked_fill(neg_violation, -1.0)
            best_idx = int(torch.argmax(strengths).item())
            best_val = float(strengths[best_idx].item())
            if best_val >= pos_hit_thr:
                selected.add(best_idx)

    if len(selected) > 0:
        best = torch.any(masks_bin[list(selected)], dim=0)
    else:
        if len(pos_xs) > 0:
            pos_hits = torch.zeros((masks_bin.shape[0],), dtype=torch.float32)
            for x, y in zip(pos_xs, pos_ys):
                pos_hits += (_neigh_max(src, x, y, hit_radius) > pos_hit_thr).float()
        else:
            pos_hits = torch.zeros((masks_bin.shape[0],), dtype=torch.float32)

        if len(neg_xs) > 0:
            neg_hits = torch.zeros((masks_bin.shape[0],), dtype=torch.float32)
            for x, y in zip(neg_xs, neg_ys):
                neg_hits += (_neigh_max(src, x, y, hit_radius) > neg_hit_thr).float()
        else:
            neg_hits = torch.zeros((masks_bin.shape[0],), dtype=torch.float32)

        eligible = (pos_hits > 0) & (neg_hits == 0) if len(pos_xs) > 0 else (neg_hits == 0)
        if bool(torch.any(eligible).item()):
            best = torch.any(masks_bin[eligible], dim=0)
        else:
            score_term = torch.zeros((masks_bin.shape[0],), dtype=torch.float32)
            if scores is not None and getattr(scores, "numel", lambda: 0)() > 0:
                scores = scores.detach().to("cpu").float()
                if scores.ndim != 0:
                    score_term = scores[: masks_bin.shape[0]]
            combined = pos_hits - (neg_hits * 2.0) + (score_term * 0.01)
            best_idx = int(torch.argmax(combined).item())
            best = masks_bin[best_idx]

    if getattr(best, "ndim", 0) == 4 and best.shape[0] == 1 and best.shape[1] == 1:
        best = best[0, 0]
    elif getattr(best, "ndim", 0) == 3 and best.shape[0] == 1:
        best = best[0]
    elif getattr(best, "ndim", 0) > 2:
        best = best.squeeze()
    mask_u8 = (best.numpy().astype(np.uint8) * 255).astype(np.uint8)
    return mask_u8


def decode_data_url_to_pil(data_url: str) -> Tuple[Image.Image, Image.Image | None]:
    if not data_url or "," not in str(data_url):
        raise ValueError("Invalid image data URL")

    raw_b64 = str(data_url).split(",", 1)[1]
    img_data = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(img_data))

    alpha = None
    if img.mode == "RGBA":
        alpha = img.split()[3]
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img, alpha


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def mask_u8_to_data_url(mask_u8: np.ndarray) -> str:
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be 2D")
    img = Image.fromarray(mask_u8.astype(np.uint8), mode="L")
    return pil_to_data_url(img)


def fill_mask_holes(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8.ndim != 2:
        return mask_u8
    try:
        import cv2  # noqa: E402

        m = (mask_u8 > 127).astype(np.uint8) * 255
        inv = 255 - m
        padded = np.pad(inv, ((1, 1), (1, 1)), mode="constant", constant_values=255)
        h, w = padded.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(padded, flood_mask, (0, 0), 0)
        holes = padded[1:-1, 1:-1]
        filled = np.clip(m + holes, 0, 255).astype(np.uint8)
        return filled
    except Exception:
        return mask_u8


def close_mask(mask_u8: np.ndarray, *, radius: int) -> np.ndarray:
    if mask_u8.ndim != 2:
        return mask_u8
    r = int(radius)
    if r <= 0:
        return mask_u8
    try:
        import cv2  # noqa: E402

        m = (mask_u8 > 127).astype(np.uint8) * 255
        k = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return closed.astype(np.uint8)
    except Exception:
        return mask_u8


def compute_point_hits(
    mask_u8: np.ndarray,
    *,
    positive_points: Iterable[dict[str, Any]] | None,
    negative_points: Iterable[dict[str, Any]] | None,
    radius: int = 2,
) -> tuple[list[bool], list[bool]]:
    if mask_u8.ndim != 2:
        return [], []

    h, w = mask_u8.shape
    r = max(0, int(radius))
    pos = _normalize_points(positive_points)
    neg = _normalize_points(negative_points)

    def _hit_foreground(x01: float, y01: float) -> bool:
        x = int(round(float(x01) * max(1, w - 1)))
        y = int(round(float(y01) * max(1, h - 1)))
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        patch = mask_u8[y0:y1, x0:x1]
        return bool(patch.size > 0 and int(patch.max()) > 127)

    def _hit_background(x01: float, y01: float) -> bool:
        x = int(round(float(x01) * max(1, w - 1)))
        y = int(round(float(y01) * max(1, h - 1)))
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        patch = mask_u8[y0:y1, x0:x1]
        return bool(patch.size > 0 and int(patch.max()) <= 127)

    pos_hit = [_hit_foreground(x, y) for x, y in pos]
    neg_hit = [_hit_background(x, y) for x, y in neg]
    return pos_hit, neg_hit


def apply_mask_to_image(
    image_rgb: Image.Image, mask_u8: np.ndarray, *, original_alpha: Image.Image | None
) -> Image.Image:
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be 2D")
    if image_rgb.mode != "RGB":
        image_rgb = image_rgb.convert("RGB")

    h, w = mask_u8.shape
    if (w, h) != image_rgb.size:
        mask_img = Image.fromarray(mask_u8.astype(np.uint8), mode="L").resize(
            image_rgb.size, resample=Image.NEAREST
        )
        mask_u8 = np.array(mask_img, dtype=np.uint8)

    out = image_rgb.convert("RGBA")
    mask_alpha = mask_u8.astype(np.uint8)

    if original_alpha is not None:
        oa = np.array(original_alpha.resize(out.size, resample=Image.NEAREST), dtype=np.uint8)
        mask_alpha = np.minimum(mask_alpha, oa).astype(np.uint8)

    out.putalpha(Image.fromarray(mask_alpha, mode="L"))
    return out


def build_sam3_image_response(
    *,
    image_data_url: str,
    positive_points: Iterable[dict[str, Any]] | None,
    negative_points: Iterable[dict[str, Any]] | None,
    threshold: float = 0.3,
    fill_holes: bool = False,
    mask_threshold: float = 0.4,
    close_radius: int = 1,
) -> dict[str, Any]:
    image_rgb, original_alpha = decode_data_url_to_pil(image_data_url)
    mask_u8 = run_sam3_image_mask_from_points(
        image_rgb,
        positive_points=positive_points,
        negative_points=negative_points,
        threshold=threshold,
        mask_threshold=mask_threshold,
    )
    mask_u8 = close_mask(mask_u8, radius=int(close_radius))
    if bool(fill_holes):
        mask_u8 = fill_mask_holes(mask_u8)
    cutout = apply_mask_to_image(image_rgb, mask_u8, original_alpha=original_alpha)
    pos_hit, neg_hit = compute_point_hits(
        mask_u8,
        positive_points=positive_points,
        negative_points=negative_points,
        radius=2,
    )

    return {
        "mask": mask_u8_to_data_url(mask_u8),
        "cutout_image": pil_to_data_url(cutout),
        "width": int(image_rgb.width),
        "height": int(image_rgb.height),
        "pos_hit": pos_hit,
        "neg_hit": neg_hit,
        "id": uuid.uuid4().hex,
    }
