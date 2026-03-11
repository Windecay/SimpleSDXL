import base64
import io
import logging
import threading
import os
import importlib.util

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_detector_lock = threading.Lock()
_detector_cache = None
_torch_backend_lock = threading.Lock()
_torch_backend_cache = None
_hybrid_backend_lock = threading.Lock()
_hybrid_backend_cache = None


def _get_detector():
    global _detector_cache
    with _detector_lock:
        if _detector_cache is not None:
            return _detector_cache
        from extras.easy_dwpose.dwpose import DWposeDetector
        _detector_cache = DWposeDetector()
        return _detector_cache


def _get_expected_onnx_paths():
    from modules import config as modules_config
    controlnet_root = modules_config.paths_controlnet[0]
    model_root = os.path.join(controlnet_root, "yzd-v", "DWPose")
    model_det = os.path.join(model_root, "yolox_l.onnx")
    model_pose = os.path.join(model_root, "dw-ll_ucoco_384.onnx")
    return model_root, model_det, model_pose


def _get_expected_torchscript_paths():
    from modules import config as modules_config
    controlnet_root = modules_config.paths_controlnet[0]
    model_pose = os.path.join(controlnet_root, "hr16", "DWPose-TorchScript-BatchSize5", "dw-ll_ucoco_384_bs5.torchscript.pt")
    model_det = os.path.join(controlnet_root, "hr16", "yolox-onnx", "yolox_l.torchscript.pt")
    return model_det, model_pose


def _format_pose(candidates, scores, width, height):
    num_candidates, _, locs = candidates.shape
    candidates = candidates.copy()
    candidates[..., 0] /= float(width)
    candidates[..., 1] /= float(height)
    bodies = candidates[:, :18].copy()
    bodies = bodies.reshape(num_candidates * 18, locs)
    body_scores = scores[:, :18].copy()
    for i in range(len(body_scores)):
        for j in range(len(body_scores[i])):
            body_scores[i][j] = int(18 * i + j) if body_scores[i][j] > 0.3 else -1
    faces = candidates[:, 24:92]
    faces_scores = scores[:, 24:92]
    hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
    hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])
    return dict(
        bodies=bodies,
        body_scores=body_scores,
        hands=hands,
        hands_scores=hands_scores,
        faces=faces,
        faces_scores=faces_scores,
    )


def _get_torch_dwpose_backend():
    global _torch_backend_cache
    with _torch_backend_lock:
        if _torch_backend_cache is not None:
            return _torch_backend_cache

        import torch

        model_det_path, model_pose_path = _get_expected_torchscript_paths()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        det = torch.jit.load(model_det_path, map_location=device).eval()
        pose = torch.jit.load(model_pose_path, map_location=device).eval()

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dwpose_torchscript"))
        det_py = os.path.join(base_dir, "jit_det.py")
        pose_py = os.path.join(base_dir, "jit_pose.py")

        spec_det = importlib.util.spec_from_file_location("layerforge_dwpose_jit_det", det_py)
        spec_pose = importlib.util.spec_from_file_location("layerforge_dwpose_jit_pose", pose_py)
        if not spec_det or not spec_det.loader or not spec_pose or not spec_pose.loader:
            raise RuntimeError("无法加载 TorchScript DWPose 推理模块")

        mod_det = importlib.util.module_from_spec(spec_det)
        mod_pose = importlib.util.module_from_spec(spec_pose)
        spec_det.loader.exec_module(mod_det)
        spec_pose.loader.exec_module(mod_pose)

        _torch_backend_cache = {
            "device": device,
            "det": det,
            "pose": pose,
            "inference_detector": getattr(mod_det, "inference_detector"),
            "inference_pose": getattr(mod_pose, "inference_pose"),
        }
        return _torch_backend_cache


def _get_hybrid_dwpose_backend(onnx_det_path: str, torch_pose_path: str):
    global _hybrid_backend_cache
    with _hybrid_backend_lock:
        if _hybrid_backend_cache is not None:
            cached = _hybrid_backend_cache
            if cached.get("onnx_det_path") == onnx_det_path and cached.get("torch_pose_path") == torch_pose_path:
                return cached

        import torch
        import onnxruntime

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose = torch.jit.load(torch_pose_path, map_location=device).eval()

        providers = ["CPUExecutionProvider"]
        provider_options = None
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider"]
            provider_options = [{"device_id": int(device.index) if device.index is not None else 0}]

        det_session = onnxruntime.InferenceSession(
            path_or_bytes=onnx_det_path,
            providers=providers,
            provider_options=provider_options,
        )

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dwpose_torchscript"))
        pose_py = os.path.join(base_dir, "jit_pose.py")
        spec_pose = importlib.util.spec_from_file_location("layerforge_dwpose_jit_pose_hybrid", pose_py)
        if not spec_pose or not spec_pose.loader:
            raise RuntimeError("无法加载 TorchScript DWPose 推理模块")
        mod_pose = importlib.util.module_from_spec(spec_pose)
        spec_pose.loader.exec_module(mod_pose)

        from extras.easy_dwpose.body_estimation.detector import inference_detector

        _hybrid_backend_cache = {
            "onnx_det_path": onnx_det_path,
            "torch_pose_path": torch_pose_path,
            "device": device,
            "det_session": det_session,
            "pose": pose,
            "inference_detector": inference_detector,
            "inference_pose": getattr(mod_pose, "inference_pose"),
        }
        return _hybrid_backend_cache


def check_model_availability():
    try:
        ts_det, ts_pose = _get_expected_torchscript_paths()
        has_ts_det = os.path.exists(ts_det) and os.path.getsize(ts_det) > 0
        has_ts_pose = os.path.exists(ts_pose) and os.path.getsize(ts_pose) > 0
        if has_ts_det and has_ts_pose:
            return {
                "available": True,
                "reason": "ready",
                "backend": "torchscript",
                "message": "OpenPose/DWPose engine is ready (TorchScript).",
                "expected": {
                    "torchscript_det": os.path.abspath(ts_det),
                    "torchscript_pose": os.path.abspath(ts_pose),
                },
            }
        model_root, model_det, model_pose = _get_expected_onnx_paths()
        has_det = os.path.exists(model_det) and os.path.getsize(model_det) > 0
        has_pose = os.path.exists(model_pose) and os.path.getsize(model_pose) > 0
        if has_det and has_ts_pose:
            return {
                "available": True,
                "reason": "ready",
                "backend": "hybrid",
                "message": "OpenPose/DWPose engine is ready (Hybrid: ONNX det + TorchScript pose).",
                "expected": {
                    "model_root": os.path.abspath(model_root),
                    "onnx_det": os.path.abspath(model_det),
                    "torchscript_pose": os.path.abspath(ts_pose),
                    "onnx_pose": os.path.abspath(model_pose),
                    "torchscript_det": os.path.abspath(ts_det),
                },
            }
        if not (has_det and has_pose):
            return {
                "available": False,
                "reason": "not_downloaded",
                "message": "未找到 LayerForge OpenPose 所需的模型文件。",
                "expected": {
                    "model_root": os.path.abspath(model_root),
                    "model_det": os.path.abspath(model_det),
                    "model_pose": os.path.abspath(model_pose),
                    "torchscript_det": os.path.abspath(ts_det),
                    "torchscript_pose": os.path.abspath(ts_pose),
                },
            }
        return {
            "available": True,
            "reason": "ready",
            "backend": "onnx",
            "message": "OpenPose/DWPose engine is ready (ONNX).",
            "expected": {
                "model_root": os.path.abspath(model_root),
                "model_det": os.path.abspath(model_det),
                "model_pose": os.path.abspath(model_pose),
                "torchscript_det": os.path.abspath(ts_det),
                "torchscript_pose": os.path.abspath(ts_pose),
            },
        }
    except ModuleNotFoundError as e:
        return {
            "available": False,
            "reason": "missing_dependency",
            "message": f"缺少依赖: {e}",
        }
    except Exception as e:
        return {
            "available": False,
            "reason": "error",
            "message": str(e),
        }


def _data_url_to_pil(image_data_url: str) -> Image.Image:
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image"):
        raise ValueError("Missing or invalid 'image' data URL.")
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data_url)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _pil_to_png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _dwpose_to_openpose_json(pose: dict, original_width: int, original_height: int) -> dict:
    bodies = pose.get("bodies")
    body_scores = pose.get("body_scores")
    if bodies is None or body_scores is None:
        return {"width": original_width, "height": original_height, "people": []}

    bodies = np.asarray(bodies)
    body_scores = np.asarray(body_scores)
    faces = pose.get("faces")
    faces_scores = pose.get("faces_scores")
    hands = pose.get("hands")
    hands_scores = pose.get("hands_scores")
    faces = np.asarray(faces) if faces is not None else None
    faces_scores = np.asarray(faces_scores) if faces_scores is not None else None
    hands = np.asarray(hands) if hands is not None else None
    hands_scores = np.asarray(hands_scores) if hands_scores is not None else None

    img_cx = float(original_width) / 2.0
    img_cy = float(original_height) / 2.0
    diag = max(1.0, (float(original_width) ** 2 + float(original_height) ** 2) ** 0.5)

    def _flat_xyc(points_xy, points_score, w, h, thr=0.3):
        out = []
        if points_xy is None:
            return out
        pts = np.asarray(points_xy)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return out
        scores_arr = None
        if points_score is not None:
            scores_arr = np.asarray(points_score).reshape(-1)
        for k in range(pts.shape[0]):
            c = float(scores_arr[k]) if scores_arr is not None and k < scores_arr.shape[0] else 1.0
            if not (c > thr):
                out.extend([0.0, 0.0, 0.0])
                continue
            x = float(pts[k, 0]) * float(w)
            y = float(pts[k, 1]) * float(h)
            out.extend([x, y, c])
        return out

    people_with_score = []
    people_count = int(body_scores.shape[0])
    for i in range(people_count):
        keypoints_flat = []
        xs = []
        ys = []
        cs = []
        for j in range(18):
            idx = int(body_scores[i, j])
            if idx < 0 or idx >= bodies.shape[0]:
                keypoints_flat.extend([0.0, 0.0, 0.0])
                continue
            x = float(bodies[idx, 0]) * float(original_width)
            y = float(bodies[idx, 1]) * float(original_height)
            c = float(bodies[idx, 2]) if bodies.shape[1] >= 3 else 1.0
            keypoints_flat.extend([x, y, c])
            if c > 0:
                xs.append(x)
                ys.append(y)
                cs.append(c)
        score = 0.0
        if len(xs) >= 4:
            min_x = float(min(xs))
            max_x = float(max(xs))
            min_y = float(min(ys))
            max_y = float(max(ys))
            area = max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))
            avg_c = float(sum(cs)) / float(len(cs))
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
            dist_norm = min(1.0, dist / diag)
            center_factor = max(0.0, 1.0 - 0.7 * dist_norm)
            score = area * avg_c * center_factor
        person = {"pose_keypoints_2d": keypoints_flat}
        if faces is not None and i < faces.shape[0]:
            face_xy = faces[i]
            face_sc = faces_scores[i] if faces_scores is not None and i < faces_scores.shape[0] else None
            person["face_keypoints_2d"] = _flat_xyc(face_xy, face_sc, original_width, original_height)
        if hands is not None and hands.shape[0] >= (people_count * 2):
            left_xy = hands[i]
            right_xy = hands[i + people_count]
            left_sc = hands_scores[i] if hands_scores is not None and i < hands_scores.shape[0] else None
            right_sc = hands_scores[i + people_count] if hands_scores is not None and (i + people_count) < hands_scores.shape[0] else None
            person["hand_left_keypoints_2d"] = _flat_xyc(left_xy, left_sc, original_width, original_height)
            person["hand_right_keypoints_2d"] = _flat_xyc(right_xy, right_sc, original_width, original_height)
        people_with_score.append((score, person))

    people_with_score.sort(key=lambda x: x[0], reverse=True)
    people = [p for _, p in people_with_score]
    return {"width": int(original_width), "height": int(original_height), "people": people}


def process_openpose(image_data_url: str, detect_resolution: int = 512, allow_download: bool = False):
    image = _data_url_to_pil(image_data_url)
    original_width, original_height = image.size

    ts_det, ts_pose = _get_expected_torchscript_paths()
    has_ts_det = os.path.exists(ts_det) and os.path.getsize(ts_det) > 0
    has_ts_pose = os.path.exists(ts_pose) and os.path.getsize(ts_pose) > 0
    if allow_download and not has_ts_pose:
        try:
            from modules.model_loader import load_file_from_url
            ts_pose_dir = os.path.dirname(ts_pose)
            os.makedirs(ts_pose_dir, exist_ok=True)
            load_file_from_url(
                url="https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt",
                model_dir=ts_pose_dir,
                file_name=os.path.basename(ts_pose),
            )
        except Exception:
            pass
        has_ts_det = os.path.exists(ts_det) and os.path.getsize(ts_det) > 0
        has_ts_pose = os.path.exists(ts_pose) and os.path.getsize(ts_pose) > 0

    if has_ts_det and has_ts_pose:
        backend = _get_torch_dwpose_backend()
        image_np = np.array(image).copy()
        height, width, _ = image_np.shape
        out_bbox = backend["inference_detector"](backend["det"], image_np, detect_classes=[0])
        if out_bbox is None:
            out_bbox = []
        keypoints, scores = backend["inference_pose"](backend["pose"], out_bbox, image_np)
        if keypoints is None or scores is None:
            pose_json_obj = {"width": int(original_width), "height": int(original_height), "people": []}
            return {"pose_json": pose_json_obj, "pose_image": _pil_to_png_data_url(Image.new("RGB", (original_width, original_height)))}
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        keypoints_info[:, openpose_idx] = keypoints_info[:, mmpose_idx]
        candidates, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        pose = _format_pose(candidates, scores, width, height)
        from extras.easy_dwpose.draw import draw_openpose
        pose_json_obj = _dwpose_to_openpose_json(pose, original_width, original_height)
        pose_image_np = draw_openpose(pose, height=height, width=width)
        pose_image_pil = Image.fromarray(np.asarray(pose_image_np)).resize((original_width, original_height), Image.LANCZOS)
        return {
            "pose_json": pose_json_obj,
            "pose_image": _pil_to_png_data_url(pose_image_pil),
        }

    model_root, model_det, model_pose = _get_expected_onnx_paths()
    has_det = os.path.exists(model_det) and os.path.getsize(model_det) > 0
    has_pose = os.path.exists(model_pose) and os.path.getsize(model_pose) > 0
    if allow_download and not has_det:
        try:
            from modules.model_loader import load_file_from_url
            os.makedirs(model_root, exist_ok=True)
            load_file_from_url(
                url="https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
                model_dir=model_root,
                file_name=os.path.basename(model_det),
            )
        except Exception:
            pass
        has_det = os.path.exists(model_det) and os.path.getsize(model_det) > 0

    if has_det and has_ts_pose:
        backend = _get_hybrid_dwpose_backend(model_det, ts_pose)
        image_np = np.array(image).copy()
        height, width, _ = image_np.shape
        out_bbox = backend["inference_detector"](backend["det_session"], image_np)
        if out_bbox is None:
            out_bbox = []
        keypoints, scores = backend["inference_pose"](backend["pose"], out_bbox, image_np)
        if keypoints is None or scores is None:
            pose_json_obj = {"width": int(original_width), "height": int(original_height), "people": []}
            return {"pose_json": pose_json_obj, "pose_image": _pil_to_png_data_url(Image.new("RGB", (original_width, original_height)))}
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        keypoints_info[:, openpose_idx] = keypoints_info[:, mmpose_idx]
        candidates, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        pose = _format_pose(candidates, scores, width, height)
        from extras.easy_dwpose.draw import draw_openpose
        pose_json_obj = _dwpose_to_openpose_json(pose, original_width, original_height)
        pose_image_np = draw_openpose(pose, height=height, width=width)
        pose_image_pil = Image.fromarray(np.asarray(pose_image_np)).resize((original_width, original_height), Image.LANCZOS)
        return {
            "pose_json": pose_json_obj,
            "pose_image": _pil_to_png_data_url(pose_image_pil),
        }
    if allow_download and not (has_det and has_pose):
        try:
            from modules.config import downloading_controlnet_dwpose
            downloading_controlnet_dwpose()
        except Exception:
            pass
        has_det = os.path.exists(model_det) and os.path.getsize(model_det) > 0
        has_pose = os.path.exists(model_pose) and os.path.getsize(model_pose) > 0

    if not (has_det and has_pose):
        raise RuntimeError(
            "未找到 LayerForge OpenPose 所需的模型文件"
            f" torchscript_det: {os.path.abspath(ts_det)} ; torchscript_pose: {os.path.abspath(ts_pose)} ;"
            f" onnx_det: {os.path.abspath(model_det)} ; onnx_pose: {os.path.abspath(model_pose)}"
        )

    detector = _get_detector()
    from extras.easy_dwpose.body_estimation import resize_image
    from extras.easy_dwpose.draw import draw_openpose

    image_np = np.array(image).copy()
    resized = resize_image(image_np, target_resolution=int(detect_resolution))
    height, width, _ = resized.shape

    candidates, scores = detector.pose_estimation(resized)
    pose = detector._format_pose(candidates, scores, width, height)

    pose_json_obj = _dwpose_to_openpose_json(pose, original_width, original_height)
    pose_image_np = draw_openpose(pose, height=height, width=width)
    pose_image_np = np.array(Image.fromarray(pose_image_np).resize((original_width, original_height), Image.LANCZOS))
    pose_image_pil = Image.fromarray(pose_image_np)

    return {
        "pose_json": pose_json_obj,
        "pose_image": _pil_to_png_data_url(pose_image_pil),
    }
