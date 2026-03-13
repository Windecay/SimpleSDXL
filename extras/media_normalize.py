import os
import base64
import tempfile
import wave
import numpy as np
import gradio as gr


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


def normalize_gradio_audio_value(audio):
    if audio is None:
        return None
    if isinstance(audio, str):
        p = audio.strip()
        return p if p else None
    if isinstance(audio, dict):
        if "waveform" in audio and "sample_rate" in audio:
            return _write_wav_temp(audio.get("sample_rate", None), audio.get("waveform", None))
        for k in ("path", "name"):
            p = audio.get(k, None)
            if isinstance(p, str) and p.strip():
                p2 = p.strip()
                if os.path.exists(p2):
                    return p2
        data = audio.get("data", None)
        name = audio.get("name", None)
        suffix = ""
        if isinstance(name, str):
            _, ext = os.path.splitext(name)
            suffix = ext or ""
        if isinstance(data, bytes):
            return _write_bytes_temp(data, suffix or ".wav")
        if isinstance(data, str) and data.strip():
            s = data.strip()
            try:
                if s.startswith("data:") and "," in s:
                    s = s.split(",", 1)[1]
                raw = base64.b64decode(s, validate=False)
                return _write_bytes_temp(raw, suffix or ".wav")
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
