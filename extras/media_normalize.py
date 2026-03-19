import os
import base64
import tempfile
import wave
import shutil
import subprocess
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image


def patch_gradio_processing_utils_for_missing_ffprobe():
    try:
        import gradio.processing_utils as _gr_processing_utils

        try:
            from gradio.components.audio import Audio as _GradioAudioComponent

            _orig_audio_preprocess = _GradioAudioComponent.preprocess

            def _audio_preprocess_passthrough(self, x):
                if x is None:
                    return x
                try:
                    file_name, file_data, is_file = (
                        x["name"],
                        x["data"],
                        x.get("is_file", False),
                    )
                    crop_min, crop_max = x.get("crop_min", 0), x.get("crop_max", 100)
                except Exception:
                    return _orig_audio_preprocess(self, x)

                try:
                    from gradio_client import utils as client_utils
                except Exception:
                    client_utils = None

                try:
                    if is_file:
                        if client_utils is not None and client_utils.is_http_url_like(file_name):
                            temp_file_path = self.download_temp_copy_if_needed(file_name)
                        else:
                            temp_file_path = self.make_temp_copy_if_needed(file_name)
                    else:
                        temp_file_path = self.base64_to_temp_file_if_needed(file_data, file_name)
                except Exception:
                    return _orig_audio_preprocess(self, x)

                if getattr(self, "type", None) == "filepath" and crop_min == 0 and crop_max == 100:
                    return temp_file_path

                try:
                    sample_rate, data = _gr_processing_utils.audio_from_file(
                        temp_file_path, crop_min=crop_min, crop_max=crop_max
                    )
                    temp_file_path_p = Path(temp_file_path)
                    output_file_name = str(
                        temp_file_path_p.with_name(
                            f"{temp_file_path_p.stem}-{crop_min}-{crop_max}{temp_file_path_p.suffix}"
                        )
                    )
                    if getattr(self, "type", None) == "numpy":
                        return sample_rate, data
                    if getattr(self, "type", None) == "filepath":
                        fmt = getattr(self, "format", "wav")
                        output_file = str(Path(output_file_name).with_suffix(f".{fmt}"))
                        _gr_processing_utils.audio_to_file(sample_rate, data, output_file, format=fmt)
                        return output_file
                except Exception:
                    return _orig_audio_preprocess(self, x)

                return _orig_audio_preprocess(self, x)

            _GradioAudioComponent.preprocess = _audio_preprocess_passthrough
        except Exception:
            pass

        if hasattr(_gr_processing_utils, "video_is_playable"):
            _orig_video_is_playable = _gr_processing_utils.video_is_playable

            def _video_is_playable_safe(video):
                try:
                    return _orig_video_is_playable(video)
                except Exception as e:
                    if e.__class__.__name__ == "FFExecutableNotFoundError" or "ffprobe" in str(e).lower():
                        return True
                    raise

            _gr_processing_utils.video_is_playable = _video_is_playable_safe

        if hasattr(_gr_processing_utils, "audio_is_playable"):
            _orig_audio_is_playable = _gr_processing_utils.audio_is_playable

            def _audio_is_playable_safe(audio):
                try:
                    return _orig_audio_is_playable(audio)
                except Exception as e:
                    if e.__class__.__name__ == "FFExecutableNotFoundError" or "ffprobe" in str(e).lower():
                        return True
                    raise

            _gr_processing_utils.audio_is_playable = _audio_is_playable_safe

        if hasattr(_gr_processing_utils, "audio_from_file"):
            _orig_audio_from_file = _gr_processing_utils.audio_from_file

            def _load_wav_wave_module(path: str):
                with wave.open(path, "rb") as wf:
                    sr = int(wf.getframerate())
                    channels = int(wf.getnchannels())
                    sampwidth = int(wf.getsampwidth())
                    frames = int(wf.getnframes())
                    raw = wf.readframes(frames)
                if sampwidth == 1:
                    audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    audio = (audio - 128.0) / 128.0
                elif sampwidth == 2:
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported WAV sampwidth: {sampwidth}")
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                else:
                    audio = audio.reshape(-1, 1)
                return sr, audio

            def _audio_from_file_safe(filename, *args, **kwargs):
                try:
                    return _orig_audio_from_file(filename, *args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if "ffprobe" not in msg or "not found" not in msg:
                        raise

                    wav_path = None
                    try:
                        wav_path = _transcode_audio_to_wav(filename)
                    except Exception:
                        wav_path = None

                    if isinstance(wav_path, str) and wav_path and os.path.exists(wav_path):
                        try:
                            return _orig_audio_from_file(wav_path, *args, **kwargs)
                        except Exception:
                            return _load_wav_wave_module(wav_path)

                    if isinstance(filename, str) and filename.lower().endswith(".wav") and os.path.exists(filename):
                        return _load_wav_wave_module(filename)
                    raise

            _gr_processing_utils.audio_from_file = _audio_from_file_safe
        return True
    except Exception:
        return False


def normalize_gradio_file_value(v):
    if v is None:
        return None
    if isinstance(v, str):
        p = v.strip()
        return p if p else None
    if isinstance(v, dict):
        for k in ("path", "name", "orig_name", "filename", "file"):
            p = v.get(k, None)
            if isinstance(p, str) and p.strip():
                p2 = p.strip()
                if os.path.exists(p2):
                    return p2
        return v
    return v


def _write_bytes_temp(data: bytes, suffix: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or "") as tmp:
            out_path = os.path.abspath(tmp.name)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path
    except Exception:
        return None


def _write_wav_temp(sample_rate: int, wav_data):
    if sample_rate is None or wav_data is None:
        return None
    try:
        sr = int(sample_rate)
    except Exception:
        return None
    try:
        audio = wav_data
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio)
        audio = np.squeeze(audio)
        if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > 8:
            audio = audio.T
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.dtype != np.int16:
            audio_f = audio.astype(np.float32, copy=False)
            audio_f = np.clip(audio_f, -1.0, 1.0)
            audio = (audio_f * 32767.0).astype(np.int16)
        audio = np.ascontiguousarray(audio)
        channels = int(audio.shape[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            out_path = os.path.abspath(tmp.name)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return out_path
    except Exception:
        return None


_AUDIO_COPY_PREFIX = "simpleai_audio_copy_"
_AUDIO_TRANSCODE_PREFIX = "simpleai_audio_wav_"


def _copy_existing_media_to_temp(src_path: str, prefix: str):
    if not isinstance(src_path, str):
        return None
    p = src_path.strip()
    if not p or not os.path.exists(p):
        return None
    base = os.path.basename(p)
    if prefix and base.startswith(prefix):
        return os.path.abspath(p)
    try:
        _root, ext = os.path.splitext(base)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or "", prefix=prefix or "") as tmp:
            out_path = os.path.abspath(tmp.name)
        shutil.copyfile(p, out_path)
        return out_path
    except Exception:
        return os.path.abspath(p)


def _get_ffmpeg_exe():
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if isinstance(exe, str) and exe.strip() and os.path.exists(exe.strip()):
            return exe.strip()
    except Exception:
        pass
    try:
        return shutil.which("ffmpeg")
    except Exception:
        return None


def _transcode_audio_to_wav(src_path: str, sample_rate: int = 16000, channels: int = 1):
    if not isinstance(src_path, str):
        return None
    p = src_path.strip()
    if not p or not os.path.exists(p):
        return None

    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=_AUDIO_TRANSCODE_PREFIX) as tmp:
            out_path = os.path.abspath(tmp.name)
    except Exception:
        return None

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        os.path.abspath(p),
        "-vn",
        "-ac",
        str(int(channels)),
        "-ar",
        str(int(sample_rate)),
        "-acodec",
        "pcm_s16le",
        out_path,
    ]

    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if completed.returncode == 0 and os.path.exists(out_path):
            return out_path
    except Exception:
        pass

    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass
    return None


def _has_ffprobe():
    try:
        return bool(shutil.which("ffprobe"))
    except Exception:
        return False


def _should_downsample_wav(path: str, size_threshold_mb: float = 12.0):
    try:
        if not isinstance(path, str) or not path.strip():
            return False
        p = path.strip()
        if not os.path.exists(p):
            return False
        size_b = os.path.getsize(p)
        return size_b >= int(float(size_threshold_mb) * 1024.0 * 1024.0)
    except Exception:
        return False


def normalize_gradio_audio_value(audio, copy_existing=True):
    if audio is None:
        return None
    if isinstance(audio, str):
        p = audio.strip()
        if not p:
            return None
        if p.startswith("data:") and "," in p:
            try:
                raw = base64.b64decode(p.split(",", 1)[1], validate=False)
                return _write_bytes_temp(raw, ".wav")
            except Exception:
                return None
        if not os.path.exists(p):
            return None
        _root, ext = os.path.splitext(p)
        ext_l = ext.lower()
        if ext_l != ".wav":
            if not _has_ffprobe():
                wav_path = _transcode_audio_to_wav(p)
                if wav_path:
                    return wav_path
        else:
            if not _has_ffprobe() and _should_downsample_wav(p, size_threshold_mb=12.0):
                wav_path = _transcode_audio_to_wav(p)
                if wav_path:
                    return wav_path
        if copy_existing:
            copied = _copy_existing_media_to_temp(p, _AUDIO_COPY_PREFIX)
            return copied if copied else os.path.abspath(p)
        return p
    if isinstance(audio, dict):
        if "waveform" in audio and "sample_rate" in audio:
            return _write_wav_temp(audio.get("sample_rate", None), audio.get("waveform", None))
        for k in ("path", "name"):
            p = audio.get(k, None)
            if isinstance(p, str) and p.strip():
                p2 = p.strip()
                if os.path.exists(p2):
                    _root, ext = os.path.splitext(p2)
                    ext_l = ext.lower()
                    if ext_l != ".wav":
                        if not _has_ffprobe():
                            wav_path = _transcode_audio_to_wav(p2)
                            if wav_path:
                                return wav_path
                    else:
                        if not _has_ffprobe() and _should_downsample_wav(p2, size_threshold_mb=12.0):
                            wav_path = _transcode_audio_to_wav(p2)
                            if wav_path:
                                return wav_path
                    if copy_existing:
                        copied = _copy_existing_media_to_temp(p2, _AUDIO_COPY_PREFIX)
                        return copied if copied else p2
                    return p2
        data = audio.get("data", None)
        name = audio.get("name", None)
        suffix = ""
        if isinstance(name, str):
            _, ext = os.path.splitext(name)
            suffix = ext or ""
        if isinstance(data, bytes):
            tmp_path = _write_bytes_temp(data, suffix or ".wav")
            if tmp_path and os.path.exists(tmp_path):
                _root, ext = os.path.splitext(tmp_path)
                ext_l = ext.lower()
                if ext_l != ".wav":
                    if not _has_ffprobe():
                        wav_path = _transcode_audio_to_wav(tmp_path)
                        if wav_path:
                            return wav_path
                else:
                    if not _has_ffprobe() and _should_downsample_wav(tmp_path, size_threshold_mb=12.0):
                        wav_path = _transcode_audio_to_wav(tmp_path)
                        if wav_path:
                            return wav_path
            return tmp_path
        if isinstance(data, str) and data.strip():
            s = data.strip()
            try:
                if s.startswith("data:") and "," in s:
                    s = s.split(",", 1)[1]
                raw = base64.b64decode(s, validate=False)
                tmp_path = _write_bytes_temp(raw, suffix or ".wav")
                if tmp_path and os.path.exists(tmp_path):
                    _root, ext = os.path.splitext(tmp_path)
                    ext_l = ext.lower()
                    if ext_l != ".wav":
                        if not _has_ffprobe():
                            wav_path = _transcode_audio_to_wav(tmp_path)
                            if wav_path:
                                return wav_path
                    else:
                        if not _has_ffprobe() and _should_downsample_wav(tmp_path, size_threshold_mb=12.0):
                            wav_path = _transcode_audio_to_wav(tmp_path)
                            if wav_path:
                                return wav_path
                return tmp_path
            except Exception:
                return None
        return None
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        sr, wav_data = audio
        return _write_wav_temp(sr, wav_data)
    return None


def stash_scene_media_before_generation(v, a, v_orig, state, video_key="scene_video", audio_key="scene_audio"):
    v_norm = normalize_gradio_file_value(v)
    v_orig_norm = normalize_gradio_file_value(v_orig)
    a_norm = normalize_gradio_audio_value(a)
    disvisible = state.get("scene_frontend", {}).get("disvisible", []) if isinstance(state, dict) else []
    show_video_placeholder = bool(v_norm and video_key not in disvisible)
    show_audio_placeholder = bool(a_norm and audio_key not in disvisible)
    return (
        v_norm,
        a_norm,
        v_orig_norm,
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        None,
        gr.update(visible=show_video_placeholder),
        gr.update(visible=show_audio_placeholder),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        None,
    )


def stash_scene_media_preview(v, a, v_orig):
    v_norm = normalize_gradio_file_value(v)
    v_orig_norm = normalize_gradio_file_value(v_orig)
    a_norm = normalize_gradio_audio_value(a)
    return (
        v_norm,
        a_norm,
        v_orig_norm,
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        None,
        gr.update(visible=True if v_norm else False),
        gr.update(visible=True if a_norm else False),
    )


def restore_scene_media_after_generation(state, v_bak, a_bak, v_orig_bak, video_key="scene_video", audio_key="scene_audio"):
    disvisible = state.get("scene_frontend", {}).get("disvisible", []) if isinstance(state, dict) else []
    v_norm = normalize_gradio_file_value(v_bak)
    a_norm = normalize_gradio_audio_value(a_bak)
    v_orig_norm = normalize_gradio_file_value(v_orig_bak)
    return (
        gr.update(value=v_norm, visible=video_key not in disvisible),
        gr.update(value=a_norm, visible=audio_key not in disvisible),
        v_orig_norm,
        gr.update(visible=False),
        gr.update(visible=False),
    )


def stash_preview_image(img, max_side=1280, trigger_side=2048):
    if img is None:
        return None, None
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        if max(h, w) <= trigger_side:
            return img, img
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR)
        preview = np.array(pil_img)
        return preview, img
    return img, img


def stash_preview_image_only(img, max_side=1280, trigger_side=2048):
    preview, _full = stash_preview_image(img, max_side=max_side, trigger_side=trigger_side)
    return preview


def _resize_np(img, new_w, new_h, resample):
    if img is None:
        return None
    if not isinstance(img, np.ndarray):
        return img
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((int(new_w), int(new_h)), resample=resample)
    return np.array(pil_img)


def stash_preview_sketch(sketch, max_side=1280, trigger_side=2048):
    if sketch is None:
        return None, None
    if not isinstance(sketch, dict):
        preview, full = stash_preview_image(sketch, max_side=max_side, trigger_side=trigger_side)
        return preview, full
    img = sketch.get("image", None)
    if not isinstance(img, np.ndarray):
        return None, None
    h, w = img.shape[:2]
    if max(h, w) <= trigger_side:
        return img, img
    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    preview_img = _resize_np(img, new_w, new_h, Image.Resampling.BILINEAR)
    return preview_img, img


def compose_full_sketch(preview_sketch, full_image):
    if full_image is None and isinstance(preview_sketch, dict):
        full_image = preview_sketch.get("image", None)
    if full_image is None and isinstance(preview_sketch, np.ndarray):
        full_image = preview_sketch
    if not isinstance(full_image, np.ndarray):
        return None

    mask = None
    if isinstance(preview_sketch, dict):
        mask = preview_sketch.get("mask", None)
    if isinstance(mask, np.ndarray):
        h, w = full_image.shape[:2]
        mask = _resize_np(mask, w, h, Image.Resampling.NEAREST)

    return {"image": full_image, "mask": mask}


def compose_full_mask(mask_preview, full_image):
    if mask_preview is None:
        return None
    if isinstance(mask_preview, dict):
        mask_preview = mask_preview.get("image", None) or mask_preview.get("mask", None)
    if not isinstance(mask_preview, np.ndarray):
        return None
    if isinstance(full_image, np.ndarray):
        h, w = full_image.shape[:2]
        return _resize_np(mask_preview, w, h, Image.Resampling.NEAREST)
    return mask_preview
