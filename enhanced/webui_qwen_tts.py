import os
import torch
import torchaudio
import numpy as np
import threading
import queue
from typing import Optional, List, Dict, Any, Tuple
import time
import logging
import datetime
import random
import wave
import re

# Try importing Qwen3-TTS nodes from ComfyUI custom nodes
# Assuming we are running in the root context where 'comfy' package is accessible
# or sys.path is already set up to include ComfyUI root.
try:
    from comfy.custom_nodes.ComfyUI_Qwen_TTS.nodes import (
        VoiceDesignNode,
        VoiceCloneNode,
        CustomVoiceNode,
        VoiceClonePromptNode,
        RoleBankNode,
        DialogueInferenceNode,
        SaveVoiceNode,
        LoadSpeakerNode,
        QwenTTSConfigNode,
        LANGUAGE_MAP,
        load_qwen_model
    )
except ImportError:
    # Fallback import strategy if direct import fails (e.g., path issues)
    import sys
    import importlib.util
    
    # Assuming this file is in modules/enhanced/ or enhanced/
    # And webui.py is in root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find root dir (where comfy folder is)
    # If in enhanced/, root is one level up
    root_dir = os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "enhanced" and os.path.basename(os.path.dirname(current_dir)) == "modules":
         root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    comfy_dir = os.path.join(root_dir, "comfy")
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)

    qwen_node_path = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-Qwen-TTS")
    if qwen_node_path not in sys.path:
        sys.path.insert(0, qwen_node_path)
        
    # Mock folder_paths if missing (WebUI context vs ComfyUI context)
    try:
        import folder_paths
    except ImportError:
        # Create a mock folder_paths module
        import types
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = os.path.join(root_dir, "models") if os.path.exists(os.path.join(root_dir, "models")) else os.path.join(root_dir, "SimpleModels") 
        folder_paths.base_path = root_dir
        folder_paths.get_folder_paths = lambda x: [os.path.join(folder_paths.models_dir, x)]
        folder_paths.get_filename_list = lambda x: []
        folder_paths.add_model_folder_path = lambda x, y: None
        sys.modules["folder_paths"] = folder_paths

    try:
        from nodes import (
            VoiceDesignNode,
            VoiceCloneNode,
            CustomVoiceNode,
            VoiceClonePromptNode,
            RoleBankNode,
            DialogueInferenceNode,
            SaveVoiceNode,
            LoadSpeakerNode,
            QwenTTSConfigNode,
            LANGUAGE_MAP,
            load_qwen_model
        )
    except ImportError as e:
        print(f"Warning: Failed to import Qwen-TTS nodes: {e}")
        # Mock classes for development if import fails
        class VoiceDesignNode: pass
        class VoiceCloneNode: pass
        class CustomVoiceNode: pass
        class VoiceClonePromptNode: pass
        class RoleBankNode: pass
        class DialogueInferenceNode: pass
        class SaveVoiceNode: pass
        class LoadSpeakerNode: pass
        class QwenTTSConfigNode: pass
        LANGUAGE_MAP = {"Auto": "auto"}
        def load_qwen_model(*args, **kwargs): pass

def _try_load_extra_model_paths():
    try:
        import folder_paths
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        extra_model_paths_config_path = os.path.join(comfy_root, "extra_model_paths.yaml")
        if not os.path.isfile(extra_model_paths_config_path):
            return
        try:
            from utils.extra_config import load_extra_path_config
        except Exception:
            from comfy.utils.extra_config import load_extra_path_config
        load_extra_path_config(extra_model_paths_config_path)
    except Exception:
        return

_try_load_extra_model_paths()

def synchronized_execution(func):
    return func

def enqueue_task(func, *args, **kwargs):
    try:
        import modules.async_worker as worker
    except Exception:
        return func(*args, **kwargs)

    called = False
    try:
        with worker.external_exclusive_task():
            called = True
            return func(*args, **kwargs)
    except Exception:
        if called:
            raise
        return func(*args, **kwargs)

def unload_qwen_tts_models():
    try:
        import sys
        import importlib
        mod_name = getattr(load_qwen_model, "__module__", None)
        if mod_name:
            mod = sys.modules.get(mod_name)
            if mod is None:
                mod = importlib.import_module(mod_name)
            unload_fn = getattr(mod, "unload_cached_model", None)
            if callable(unload_fn):
                unload_fn()
    except Exception:
        return
    try:
        import gc
        gc.collect()
        gc.collect()
    except Exception:
        pass
    try:
        import ldm_patched.modules.model_management as _mm
        _mm.soft_empty_cache(True)
    except Exception:
        pass
    try:
        from comfy import model_management as _cm
        _cm.soft_empty_cache(True)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# --- Qwen-TTS Feature Wrappers ---

class QwenTTSWrapper:
    def __init__(self):
        self.loaded_model = None
        self.model_config = {}

    def _parse_pause_markup(self, text: str) -> List[Tuple[str, Any]]:
        if text is None:
            return []
        s = str(text)
        if not s:
            return []

        pause_re = re.compile(r"\[\s*pause\s*=\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*\]", re.IGNORECASE)
        out: List[Tuple[str, Any]] = []
        last = 0
        for m in pause_re.finditer(s):
            start, end = m.span()
            if start > last:
                chunk = s[last:start]
                if chunk:
                    out.append(("text", chunk))
            num = m.group(1)
            unit = (m.group(2) or "ms").lower()
            try:
                v = float(num)
            except Exception:
                v = 0.0
            if unit == "s":
                seconds = v
            else:
                seconds = v / 1000.0
            seconds = float(max(0.0, min(seconds, 120.0)))
            out.append(("pause", seconds))
            last = end
        if last < len(s):
            tail = s[last:]
            if tail:
                out.append(("text", tail))
        return out

    def _expand_pause_plan(
        self,
        text: str,
        default_gap_seconds: float,
        max_chars: int = 200,
        hard_max_chars: int = 260,
    ) -> Tuple[List[Tuple[str, Any]], bool]:
        parts = self._parse_pause_markup(text)
        saw_pause = any(k == "pause" for (k, _) in parts)

        def _clamp_pause(v: float) -> float:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            return float(max(0.0, min(fv, 120.0)))

        try:
            default_gap = float(default_gap_seconds)
        except Exception:
            default_gap = 0.0
        default_gap = float(max(0.0, min(default_gap, 10.0)))

        if not saw_pause:
            segs = self._split_long_text(str(text), max_chars=max_chars, hard_max_chars=hard_max_chars)
            plan: List[Tuple[str, Any]] = []
            kept = [s for s in segs if s and str(s).strip()]
            for i, seg in enumerate(kept):
                plan.append(("text", seg))
                if default_gap > 0.0 and i != len(kept) - 1:
                    plan.append(("pause", default_gap))
            return plan, False

        plan2: List[Tuple[str, Any]] = []
        pending_pause = 0.0
        emitted_any_text = False

        for kind, payload in parts:
            if kind == "pause":
                pending_pause = _clamp_pause(pending_pause + _clamp_pause(float(payload)))
                continue

            raw = "" if payload is None else str(payload)
            segs = self._split_long_text(raw, max_chars=max_chars, hard_max_chars=hard_max_chars)
            kept = [s for s in segs if s and str(s).strip()]
            if not kept:
                continue

            if pending_pause > 0.0:
                plan2.append(("pause", pending_pause))
                pending_pause = 0.0
            elif emitted_any_text and default_gap > 0.0:
                plan2.append(("pause", default_gap))

            for i, seg in enumerate(kept):
                plan2.append(("text", seg))
                emitted_any_text = True
                if default_gap > 0.0 and i != len(kept) - 1:
                    plan2.append(("pause", default_gap))

        if pending_pause > 0.0:
            plan2.append(("pause", pending_pause))

        return plan2, True

    def _split_text_with_pause_markup(self, text: str, max_chars: int = 200, hard_max_chars: int = 260) -> List[Tuple[str, Any]]:
        parts = self._parse_pause_markup(text)
        if not parts:
            return [("text", x) for x in self._split_long_text(text, max_chars=max_chars, hard_max_chars=hard_max_chars)]
        out: List[Tuple[str, Any]] = []
        for kind, payload in parts:
            if kind == "pause":
                out.append((kind, payload))
                continue
            for seg in self._split_long_text(str(payload), max_chars=max_chars, hard_max_chars=hard_max_chars):
                if seg and str(seg).strip():
                    out.append(("text", seg))
        merged: List[Tuple[str, Any]] = []
        pending_pause = 0.0
        for kind, payload in out:
            if kind == "pause":
                try:
                    pending_pause += float(payload)
                except Exception:
                    pending_pause += 0.0
                continue
            if pending_pause > 0.0:
                merged.append(("pause", pending_pause))
                pending_pause = 0.0
            merged.append(("text", payload))
        if pending_pause > 0.0:
            merged.append(("pause", pending_pause))
        return merged

    def _make_silence_audio_dict(self, sr: int, seconds: float, target_audio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sample_rate = int(max(1, int(sr)))
        dur = float(max(0.0, min(float(seconds), 120.0)))
        n = int(round(dur * float(sample_rate)))
        channels = 1
        dtype = torch.float32
        if isinstance(target_audio, dict):
            w = target_audio.get("waveform")
            if hasattr(w, "detach"):
                w = w.detach()
            if hasattr(w, "cpu"):
                w = w.cpu()
            if isinstance(w, np.ndarray):
                w = torch.from_numpy(w)
            if hasattr(w, "ndim") and int(getattr(w, "ndim", 0)) >= 2:
                try:
                    channels = int(w.shape[1])
                except Exception:
                    channels = 1
            if hasattr(w, "dtype"):
                dtype = w.dtype
        if n <= 0:
            n = 0
        return {"waveform": torch.zeros((1, channels, n), dtype=dtype), "sample_rate": sample_rate}

    def _split_long_text(self, text: str, max_chars: int = 200, hard_max_chars: int = 260) -> List[str]:
        if text is None:
            return []
        raw = str(text).strip()
        if not raw:
            return []
        if len(raw) <= max_chars:
            return [raw]

        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(r"[ \t]+", " ", raw)

        sentence_delims = r"([。！？!?；;…]+|(?<!\d)\.(?!\d))"
        chunks: List[str] = []
        current = ""

        def add_terminal_period(s: str) -> str:
            t = (s or "").strip()
            if not t:
                return t
            if t.endswith("。。"):
                return t + "。"
            if t.endswith(("。", "！", "？", "!", "?")):
                return t + "。。"
            return t + "。。"

        def flush():
            nonlocal current
            s = current.strip()
            if s:
                chunks.append(add_terminal_period(s))
            current = ""

        lines = [ln.strip() for ln in raw.split("\n")]
        for line in lines:
            if not line:
                flush()
                continue
            parts = re.split(sentence_delims, line)
            pieces: List[str] = []
            for i in range(0, len(parts), 2):
                base = (parts[i] or "").strip()
                if not base:
                    continue
                delim = parts[i + 1] if i + 1 < len(parts) else ""
                pieces.append((base + (delim or "")).strip())

            for piece in pieces or [line]:
                if len(piece) > hard_max_chars:
                    weak_delims = r"([，,、:：]+)"
                    weak_parts = re.split(weak_delims, piece)
                    weak_pieces: List[str] = []
                    for i in range(0, len(weak_parts), 2):
                        base = (weak_parts[i] or "").strip()
                        if not base:
                            continue
                        delim = weak_parts[i + 1] if i + 1 < len(weak_parts) else ""
                        weak_pieces.append((base + (delim or "")).strip())
                    for wp in weak_pieces or [piece]:
                        if len(wp) > hard_max_chars:
                            start = 0
                            while start < len(wp):
                                sub = wp[start:start + hard_max_chars].strip()
                                if sub:
                                    if current and len(current) + 1 + len(sub) > max_chars:
                                        flush()
                                    current = (current + " " + sub).strip() if current else sub
                                    flush()
                                start += hard_max_chars
                        else:
                            if current and len(current) + 1 + len(wp) > max_chars:
                                flush()
                            current = (current + " " + wp).strip() if current else wp
                            flush()
                    continue

                if not current:
                    current = piece
                    continue
                if len(current) + 1 + len(piece) <= max_chars:
                    current = (current + " " + piece).strip()
                else:
                    flush()
                    current = piece

        flush()
        if chunks:
            return chunks
        return [add_terminal_period(raw)]

    def _merge_audio_dicts(self, audio_dicts: List[Dict[str, Any]], gap_seconds: float = 0.06, tail_seconds: float = 0.32) -> Dict[str, Any]:
        if not audio_dicts:
            raise ValueError("No audio segments to merge")
        sr = int(audio_dicts[0].get("sample_rate"))
        waveforms = []
        for a in audio_dicts:
            if int(a.get("sample_rate")) != sr:
                raise ValueError("Mismatched sample_rate across segments")
            w = a.get("waveform")
            if hasattr(w, "detach"):
                w = w.detach()
            if hasattr(w, "cpu"):
                w = w.cpu()
            if isinstance(w, np.ndarray):
                w = torch.from_numpy(w)
            waveforms.append(w)

        target_channels = int(waveforms[0].shape[1]) if getattr(waveforms[0], "ndim", 0) >= 2 else 1
        fixed = []
        for w in waveforms:
            if getattr(w, "ndim", 0) == 1:
                w = w[None, None, :]
            elif getattr(w, "ndim", 0) == 2:
                w = w[None, :, :]
            if int(w.shape[1]) != target_channels:
                if target_channels == 2 and int(w.shape[1]) == 1:
                    w = w.repeat(1, 2, 1)
                elif target_channels == 1 and int(w.shape[1]) == 2:
                    w = w.mean(dim=1, keepdim=True)
            fixed.append(w)

        def _edge_silence_seconds(w: torch.Tensor, at_start: bool) -> float:
            try:
                x = w
                if getattr(x, "ndim", 0) == 3:
                    x = x[0]
                if getattr(x, "ndim", 0) == 2:
                    x = x.mean(dim=0)
                x = x.to(torch.float32)
                n = int(x.shape[-1])
                if n <= 0:
                    return 0.0
                edge = int(min(n, int(float(sr) * 0.8)))
                if edge <= 0:
                    return 0.0
                seg = x[:edge] if at_start else x[-edge:]
                thr = 0.0035
                mask = seg.abs() > thr
                if not bool(mask.any().item()):
                    return float(edge) / float(sr)
                if at_start:
                    first = int(torch.argmax(mask.to(torch.int32)).item())
                    return float(first) / float(sr)
                rev = torch.flip(mask, dims=[0])
                last_from_end = int(torch.argmax(rev.to(torch.int32)).item())
                return float(last_from_end) / float(sr)
            except Exception:
                return 0.0

        min_pause_s = float(max(0.0, float(gap_seconds)))
        if len(fixed) == 1 or min_pause_s <= 0.0:
            merged_waveform = torch.cat(fixed, dim=-1)
        else:
            leading = [_edge_silence_seconds(w, at_start=True) for w in fixed]
            trailing = [_edge_silence_seconds(w, at_start=False) for w in fixed]
            merged_parts = [fixed[0]]
            for i in range(len(fixed) - 1):
                pause_present = float(trailing[i]) + float(leading[i + 1])
                need = max(0.0, min_pause_s - pause_present)
                need_samples = int(need * float(sr))
                if need_samples > 0:
                    merged_parts.append(torch.zeros((1, target_channels, need_samples), dtype=fixed[i].dtype))
                merged_parts.append(fixed[i + 1])
            merged_waveform = torch.cat(merged_parts, dim=-1)
        tail_samples = int(max(0.0, float(tail_seconds)) * float(sr))
        if tail_samples > 0:
            merged_waveform = torch.cat(
                [merged_waveform, torch.zeros((1, target_channels, tail_samples), dtype=merged_waveform.dtype)],
                dim=-1,
            )
        return {"waveform": merged_waveform, "sample_rate": sr}

    def _extract_audio_dict(self, result: Any) -> Dict[str, Any]:
        if not result or not isinstance(result, tuple):
            raise ValueError("Invalid output from node")
        audio_dict = result[0]
        if not isinstance(audio_dict, dict):
            raise ValueError("Invalid audio output type")
        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            raise ValueError("Missing waveform or sample_rate")
        return audio_dict

    def _get_user_did(self, user_did: Optional[str]) -> Optional[str]:
        if user_did:
            return user_did
        try:
            import shared
            return shared.token.get_guest_did()
        except Exception:
            return None

    def _get_output_path_for_user(self, user_did: Optional[str]) -> str:
        try:
            import shared
            token = getattr(shared, "token", None)
            if token is not None and hasattr(token, "get_path_in_user_dir"):
                did = user_did or token.get_guest_did()
                user_path_outputs = token.get_path_in_user_dir(did, "outputs")
                os.makedirs(user_path_outputs, exist_ok=True)
                return user_path_outputs
        except Exception:
            pass
        try:
            import modules.config as config
            return config.get_user_path_outputs(user_did)
        except Exception:
            return os.path.abspath("./outputs")

    def _save_wav(self, sr: int, wav: Any, prefix: str, user_did: Optional[str], log_metadata: Optional[list] = None) -> str:
        user_did = self._get_user_did(user_did)
        base_dir = self._get_output_path_for_user(user_did)

        date_dir = datetime.datetime.now().strftime("%Y-%m-%d")
        time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rand = random.randint(1000, 9999)
        filename = f"{prefix}_{time_tag}_{rand}.wav"
        out_dir = os.path.join(base_dir, date_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.abspath(os.path.join(out_dir, filename))

        audio = wav
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
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(audio.tobytes())

        try:
            from modules.private_logger import log_audio_file

            meta = list(log_metadata) if isinstance(log_metadata, list) else []
            meta = [("File", "file", os.path.basename(out_path))] + meta
            meta = [("Sample Rate", "sample_rate", int(sr)), ("Channels", "channels", channels)] + meta
            log_audio_file(out_path, meta, user_did=user_did)
        except Exception:
            pass

        return out_path

    @synchronized_execution
    def voice_design(
        self,
        text,
        instruct,
        model_choice,
        user_did=None,
        device="auto",
        precision="bf16",
        language="Auto",
        seed=0,
        max_new_tokens=4096,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
        lock_timbre_with_first_segment=False,
        clone_batch_size=4,
        progress_callback=None,
    ):
        t0 = time.perf_counter()
        plan, saw_pause_markup = self._expand_pause_plan(str(text), default_gap_seconds=0.4, max_chars=200, hard_max_chars=260)
        segments = [p[1] for p in plan if p and p[0] == "text"]
        per_seg_tokens = int(max(1, int(max_new_tokens)))
        try:
            bs = int(clone_batch_size)
        except Exception:
            bs = 1
        if bs < 1:
            bs = 1
        if bs > 16:
            bs = 16
        node = VoiceDesignNode()
        audios = []
        if callable(progress_callback):
            try:
                progress_callback(0, f"准备分段：{len(segments)} 段")
            except Exception:
                pass
        lock_timbre = bool(lock_timbre_with_first_segment) and len(segments) > 1
        if not lock_timbre:
            model = load_qwen_model("VoiceDesign", model_choice, device, precision, attention, False, None, "")
            mapped_lang = LANGUAGE_MAP.get(language, "auto")
            done_count = 0
            leading_pause_s = 0.0
            i = 0
            while i < len(plan) and plan[i][0] == "pause":
                try:
                    leading_pause_s += float(plan[i][1])
                except Exception:
                    leading_pause_s += 0.0
                i += 1
            if leading_pause_s > 120.0:
                leading_pause_s = 120.0

            sr_known = None
            actions = plan[i:]
            cursor = 0
            while cursor < len(actions):
                texts: List[str] = []
                pauses_after: List[float] = []
                while cursor < len(actions) and actions[cursor][0] == "text":
                    texts.append(str(actions[cursor][1]))
                    if cursor + 1 < len(actions) and actions[cursor + 1][0] == "pause":
                        try:
                            pauses_after.append(float(actions[cursor + 1][1]))
                        except Exception:
                            pauses_after.append(0.0)
                        cursor += 2
                    else:
                        pauses_after.append(0.0)
                        cursor += 1

                if not texts:
                    cursor += 1
                    continue

                start_index = 0
                while start_index < len(texts):
                    batch_texts = texts[start_index:start_index + bs]
                    batch_pauses = pauses_after[start_index:start_index + len(batch_texts)]
                    if callable(progress_callback):
                        try:
                            pct = int(done_count * 100 / max(len(segments), 1))
                            progress_callback(pct, f"生成音频：{done_count + 1}-{min(done_count + len(batch_texts), len(segments))}/{len(segments)}")
                        except Exception:
                            pass
                    try:
                        wavs, sr = model.generate_voice_design(
                            text=batch_texts,
                            instruct=instruct,
                            language=[mapped_lang] * len(batch_texts),
                            max_new_tokens=per_seg_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ("out of memory" in msg or "cuda" in msg) and bs > 1:
                            bs = max(1, bs // 2)
                            continue
                        raise
                    for w, pause_s in zip(wavs, batch_pauses):
                        waveform = torch.from_numpy(w).float()
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0).unsqueeze(0)
                        elif waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        seg_audio = {"waveform": waveform, "sample_rate": sr}
                        if sr_known is None:
                            sr_known = int(sr)
                        if not audios and leading_pause_s > 0.0:
                            audios.append(self._make_silence_audio_dict(int(sr_known), leading_pause_s, seg_audio))
                            leading_pause_s = 0.0
                        audios.append(seg_audio)
                        if pause_s and float(pause_s) > 0.0:
                            audios.append(self._make_silence_audio_dict(int(sr_known), float(pause_s), seg_audio))
                        done_count += 1
                    start_index += len(batch_texts)
        else:
            if callable(progress_callback):
                try:
                    progress_callback(0, "生成首段（用于锁定音色）")
                except Exception:
                    pass
            leading_pause_s = 0.0
            j = 0
            while j < len(plan) and plan[j][0] == "pause":
                try:
                    leading_pause_s += float(plan[j][1])
                except Exception:
                    leading_pause_s += 0.0
                j += 1
            if leading_pause_s > 120.0:
                leading_pause_s = 120.0
            first_text = segments[0]
            first_result = node.generate(
                text=first_text,
                instruct=instruct,
                model_choice=model_choice,
                device=device,
                precision=precision,
                language=language,
                seed=int(seed),
                max_new_tokens=per_seg_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                attention=attention,
                unload_model_after_generate=False,
            )
            first_audio = self._extract_audio_dict(first_result)
            if leading_pause_s > 0.0:
                audios.append(self._make_silence_audio_dict(int(first_audio.get("sample_rate", 24000)), leading_pause_s, first_audio))
            audios.append(first_audio)

            first_text_index = None
            for idx in range(j, len(plan)):
                if plan[idx][0] == "text":
                    first_text_index = idx
                    break
            pause_after_first_s = 0.0
            if first_text_index is not None:
                k = first_text_index + 1
                while k < len(plan) and plan[k][0] == "pause":
                    try:
                        pause_after_first_s += float(plan[k][1])
                    except Exception:
                        pause_after_first_s += 0.0
                    k += 1
                if pause_after_first_s > 120.0:
                    pause_after_first_s = 120.0
                if pause_after_first_s > 0.0:
                    sr0 = int(first_audio.get("sample_rate", 24000))
                    audios.append(self._make_silence_audio_dict(sr0, pause_after_first_s, first_audio))

            if callable(progress_callback):
                try:
                    progress_callback(20, "提取首段音色特征")
                except Exception:
                    pass
            prompt_node = VoiceClonePromptNode()
            voice_clone_prompt = prompt_node.create_prompt(
                ref_audio=first_audio,
                ref_text=first_text,
                model_choice=model_choice,
                device=device,
                precision=precision,
                attention=attention,
                x_vector_only=False,
                unload_model_after_generate=False,
            )[0]

            model = load_qwen_model("Base", model_choice, device, precision, attention, False, None, "")
            mapped_lang = LANGUAGE_MAP.get(language, "auto")

            done_count = 1
            sr_known = int(first_audio.get("sample_rate", 24000))
            if first_text_index is None:
                actions = []
            else:
                k = first_text_index + 1
                while k < len(plan) and plan[k][0] == "pause":
                    k += 1
                actions = plan[k:]

            cursor = 0
            while cursor < len(actions):
                texts: List[str] = []
                pauses_after: List[float] = []
                while cursor < len(actions) and actions[cursor][0] == "text":
                    texts.append(str(actions[cursor][1]))
                    if cursor + 1 < len(actions) and actions[cursor + 1][0] == "pause":
                        try:
                            pauses_after.append(float(actions[cursor + 1][1]))
                        except Exception:
                            pauses_after.append(0.0)
                        cursor += 2
                    else:
                        pauses_after.append(0.0)
                        cursor += 1

                if not texts:
                    cursor += 1
                    continue

                start_index = 0
                while start_index < len(texts):
                    batch_texts = texts[start_index:start_index + bs]
                    batch_pauses = pauses_after[start_index:start_index + len(batch_texts)]
                    if callable(progress_callback):
                        try:
                            pct = 20 + int(done_count * 80 / max(len(segments), 1))
                            progress_callback(pct, f"锁定音色生成：{done_count + 1}-{min(done_count + len(batch_texts), len(segments))}/{len(segments)}")
                        except Exception:
                            pass
                    try:
                        wavs, sr = model.generate_voice_clone(
                            text=batch_texts,
                            language=[mapped_lang] * len(batch_texts),
                            voice_clone_prompt=voice_clone_prompt,
                            ref_text=first_text,
                            x_vector_only_mode=False,
                            max_new_tokens=per_seg_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ("out of memory" in msg or "cuda" in msg) and bs > 1:
                            bs = 1
                            continue
                        raise

                    for w, pause_s in zip(wavs, batch_pauses):
                        waveform = torch.from_numpy(w).float()
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0).unsqueeze(0)
                        elif waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        seg_audio = {"waveform": waveform, "sample_rate": sr}
                        sr_known = int(sr)
                        audios.append(seg_audio)
                        if pause_s and float(pause_s) > 0.0:
                            audios.append(self._make_silence_audio_dict(int(sr_known), float(pause_s), seg_audio))
                        done_count += 1
                    start_index += len(batch_texts)

        merge_gap = 0.0 if saw_pause_markup else 0.4
        merged = self._merge_audio_dicts(audios, gap_seconds=merge_gap) if len(audios) > 1 else audios[0]
        if unload_model_after_generate:
            try:
                unload_qwen_tts_models()
            except Exception:
                pass
        sr, wav = self._process_output((merged,))
        elapsed_s = time.perf_counter() - t0
        logging.info("QwenTTS voice_design done: batch_size=%s, elapsed_s=%.3f", bs, elapsed_s)
        if callable(progress_callback):
            try:
                progress_callback(100, f"完成，批次大小: {bs}，耗时: {elapsed_s:.3f}秒")
            except Exception:
                pass
        return self._save_wav(
            sr,
            wav,
            "tts_voice_design",
            user_did,
            log_metadata=[
                ("Mode", "mode", "voice_design"),
                ("Model", "model_choice", model_choice),
                ("Language", "language", language),
                ("Seed", "seed", seed),
                ("Text", "text", text),
                ("Instruct", "instruct", instruct),
                ("Lock Timbre", "lock_timbre_with_first_segment", lock_timbre_with_first_segment),
                ("Batch Size", "clone_batch_size", bs),
                ("Max New Tokens", "max_new_tokens", max_new_tokens),
                ("Top P", "top_p", top_p),
                ("Top K", "top_k", top_k),
                ("Temperature", "temperature", temperature),
                ("Repetition Penalty", "repetition_penalty", repetition_penalty),
                ("Elapsed(s)", "elapsed_s", f"{elapsed_s:.3f}"),
            ],
        )

    @synchronized_execution
    def voice_clone(
        self,
        ref_audio,
        ref_text,
        target_text,
        model_choice,
        user_did=None,
        device="auto",
        precision="bf16",
        language="Auto",
        seed=0,
        max_new_tokens=4096,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        x_vector_only=False,
        attention="auto",
        unload_model_after_generate=False,
        custom_model_path="",
        batch_size=4,
        progress_callback=None,
    ):
        t0 = time.perf_counter()
        plan, saw_pause_markup = self._expand_pause_plan(str(target_text), default_gap_seconds=0.14, max_chars=200, hard_max_chars=260)
        segments = [p[1] for p in plan if p and p[0] == "text"]
        per_seg_tokens = int(max(1, int(max_new_tokens)))
        prompt_node = VoiceClonePromptNode()
        audio_dict = self._audio_input_to_comfy_audio(ref_audio)
        if callable(progress_callback):
            try:
                progress_callback(0, "准备参考音频")
            except Exception:
                pass
        voice_clone_prompt = prompt_node.create_prompt(
            ref_audio=audio_dict,
            ref_text=ref_text or "",
            model_choice=model_choice,
            device=device,
            precision=precision,
            attention=attention,
            x_vector_only=bool(x_vector_only),
            unload_model_after_generate=False,
        )[0]
        if callable(progress_callback):
            try:
                progress_callback(5, f"提取声音特征，准备分段：{len(segments)} 段")
            except Exception:
                pass

        model = load_qwen_model("Base", model_choice, device, precision, attention, False, None, custom_model_path or "")
        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        try:
            bs = int(batch_size)
        except Exception:
            bs = 1
        if bs < 1:
            bs = 1
        if bs > 16:
            bs = 16

        audios = []
        done_count = 0
        sr_known = None
        leading_pause_s = 0.0
        i = 0
        while i < len(plan) and plan[i][0] == "pause":
            try:
                leading_pause_s += float(plan[i][1])
            except Exception:
                leading_pause_s += 0.0
            i += 1
        if leading_pause_s > 120.0:
            leading_pause_s = 120.0

        actions = plan[i:]
        cursor = 0
        while cursor < len(actions):
            texts: List[str] = []
            pauses_after: List[float] = []
            while cursor < len(actions) and actions[cursor][0] == "text":
                texts.append(str(actions[cursor][1]))
                if cursor + 1 < len(actions) and actions[cursor + 1][0] == "pause":
                    try:
                        pauses_after.append(float(actions[cursor + 1][1]))
                    except Exception:
                        pauses_after.append(0.0)
                    cursor += 2
                else:
                    pauses_after.append(0.0)
                    cursor += 1

            if not texts:
                cursor += 1
                continue

            start_index = 0
            while start_index < len(texts):
                batch_texts = texts[start_index:start_index + bs]
                batch_pauses = pauses_after[start_index:start_index + len(batch_texts)]
                if callable(progress_callback):
                    try:
                        pct = 5 + int(done_count * 95 / max(len(segments), 1))
                        progress_callback(pct, f"生成音频：{done_count + 1}-{min(done_count + len(batch_texts), len(segments))}/{len(segments)}")
                    except Exception:
                        pass
                try:
                    wavs, sr = model.generate_voice_clone(
                        text=batch_texts,
                        language=[mapped_lang] * len(batch_texts),
                        voice_clone_prompt=voice_clone_prompt,
                        ref_text=(ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None),
                        x_vector_only_mode=bool(x_vector_only),
                        max_new_tokens=per_seg_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda" in msg) and bs > 1:
                        bs = max(1, bs // 2)
                        continue
                    raise
                for w, pause_s in zip(wavs, batch_pauses):
                    waveform = torch.from_numpy(w).float()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                    seg_audio = {"waveform": waveform, "sample_rate": sr}
                    if sr_known is None:
                        sr_known = int(sr)
                    if not audios and leading_pause_s > 0.0:
                        audios.append(self._make_silence_audio_dict(int(sr_known), leading_pause_s, seg_audio))
                        leading_pause_s = 0.0
                    audios.append(seg_audio)
                    if pause_s and float(pause_s) > 0.0:
                        audios.append(self._make_silence_audio_dict(int(sr_known), float(pause_s), seg_audio))
                    done_count += 1
                start_index += len(batch_texts)

        merge_gap = 0.0 if saw_pause_markup else 0.14
        merged = self._merge_audio_dicts(audios, gap_seconds=merge_gap) if len(audios) > 1 else audios[0]
        if unload_model_after_generate:
            try:
                unload_qwen_tts_models()
            except Exception:
                pass
        sr, wav = self._process_output((merged,))
        elapsed_s = time.perf_counter() - t0
        logging.info("QwenTTS voice_clone done: batch_size=%s, elapsed_s=%.3f", bs, elapsed_s)
        if callable(progress_callback):
            try:
                progress_callback(100, f"完成，批次大小: {bs}，耗时: {elapsed_s:.3f}秒")
            except Exception:
                pass
        return self._save_wav(
            sr,
            wav,
            "tts_voice_clone",
            user_did,
            log_metadata=[
                ("Mode", "mode", "voice_clone"),
                ("Model", "model_choice", model_choice),
                ("Language", "language", language),
                ("Seed", "seed", seed),
                ("Target Text", "target_text", target_text),
                ("Ref Text", "ref_text", ref_text),
                ("XVector Only", "x_vector_only", x_vector_only),
                ("Batch Size", "batch_size", bs),
                ("Max New Tokens", "max_new_tokens", max_new_tokens),
                ("Top P", "top_p", top_p),
                ("Top K", "top_k", top_k),
                ("Temperature", "temperature", temperature),
                ("Repetition Penalty", "repetition_penalty", repetition_penalty),
                ("Elapsed(s)", "elapsed_s", f"{elapsed_s:.3f}"),
            ],
        )

    @synchronized_execution
    def custom_voice(
        self,
        text,
        speaker,
        model_choice,
        user_did=None,
        device="auto",
        precision="bf16",
        language="Auto",
        seed=0,
        instruct="",
        max_new_tokens=4096,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
        custom_model_path="",
        custom_speaker_name="",
        batch_size=4,
        progress_callback=None,
    ):
        t0 = time.perf_counter()
        plan, saw_pause_markup = self._expand_pause_plan(str(text), default_gap_seconds=0.14, max_chars=200, hard_max_chars=260)
        segments = [p[1] for p in plan if p and p[0] == "text"]
        per_seg_tokens = int(max(1, int(max_new_tokens)))
        model = load_qwen_model("CustomVoice", model_choice, device, precision, attention, False, None, custom_model_path or "")
        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        if custom_speaker_name and str(custom_speaker_name).strip():
            target_speaker = str(custom_speaker_name).strip()
        else:
            target_speaker = ("" if speaker is None else str(speaker)).lower().replace(" ", "_")

        try:
            bs = int(batch_size)
        except Exception:
            bs = 1
        if bs < 1:
            bs = 1
        if bs > 16:
            bs = 16

        audios = []
        if callable(progress_callback):
            try:
                progress_callback(0, f"准备分段：{len(segments)} 段")
            except Exception:
                pass
        done_count = 0
        sr_known = None
        leading_pause_s = 0.0
        i = 0
        while i < len(plan) and plan[i][0] == "pause":
            try:
                leading_pause_s += float(plan[i][1])
            except Exception:
                leading_pause_s += 0.0
            i += 1
        if leading_pause_s > 120.0:
            leading_pause_s = 120.0

        actions = plan[i:]
        cursor = 0
        while cursor < len(actions):
            texts: List[str] = []
            pauses_after: List[float] = []
            while cursor < len(actions) and actions[cursor][0] == "text":
                texts.append(str(actions[cursor][1]))
                if cursor + 1 < len(actions) and actions[cursor + 1][0] == "pause":
                    try:
                        pauses_after.append(float(actions[cursor + 1][1]))
                    except Exception:
                        pauses_after.append(0.0)
                    cursor += 2
                else:
                    pauses_after.append(0.0)
                    cursor += 1

            if not texts:
                cursor += 1
                continue

            start_index = 0
            while start_index < len(texts):
                batch_texts = texts[start_index:start_index + bs]
                batch_pauses = pauses_after[start_index:start_index + len(batch_texts)]
                if callable(progress_callback):
                    try:
                        pct = int(done_count * 100 / max(len(segments), 1))
                        progress_callback(pct, f"生成音频：{done_count + 1}-{min(done_count + len(batch_texts), len(segments))}/{len(segments)}")
                    except Exception:
                        pass
                try:
                    wavs, sr = model.generate_custom_voice(
                        text=batch_texts,
                        speaker=[target_speaker] * len(batch_texts),
                        language=[mapped_lang] * len(batch_texts),
                        instruct=(instruct if isinstance(instruct, str) and instruct.strip() else None),
                        max_new_tokens=per_seg_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda" in msg) and bs > 1:
                        bs = max(1, bs // 2)
                        continue
                    raise
                for w, pause_s in zip(wavs, batch_pauses):
                    waveform = torch.from_numpy(w).float()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                    seg_audio = {"waveform": waveform, "sample_rate": sr}
                    if sr_known is None:
                        sr_known = int(sr)
                    if not audios and leading_pause_s > 0.0:
                        audios.append(self._make_silence_audio_dict(int(sr_known), leading_pause_s, seg_audio))
                        leading_pause_s = 0.0
                    audios.append(seg_audio)
                    if pause_s and float(pause_s) > 0.0:
                        audios.append(self._make_silence_audio_dict(int(sr_known), float(pause_s), seg_audio))
                    done_count += 1
                start_index += len(batch_texts)

        merge_gap = 0.0 if saw_pause_markup else 0.14
        merged = self._merge_audio_dicts(audios, gap_seconds=merge_gap) if len(audios) > 1 else audios[0]
        if unload_model_after_generate:
            try:
                unload_qwen_tts_models()
            except Exception:
                pass
        sr, wav = self._process_output((merged,))
        elapsed_s = time.perf_counter() - t0
        logging.info("QwenTTS custom_voice done: batch_size=%s, elapsed_s=%.3f", bs, elapsed_s)
        if callable(progress_callback):
            try:
                progress_callback(100, f"完成，批次大小: {bs}，耗时: {elapsed_s:.3f}秒")
            except Exception:
                pass
        return self._save_wav(
            sr,
            wav,
            "tts_custom_voice",
            user_did,
            log_metadata=[
                ("Mode", "mode", "custom_voice"),
                ("Model", "model_choice", model_choice),
                ("Language", "language", language),
                ("Seed", "seed", seed),
                ("Speaker", "speaker", speaker),
                ("Custom Speaker", "custom_speaker_name", custom_speaker_name),
                ("Text", "text", text),
                ("Instruct", "instruct", instruct),
                ("Batch Size", "batch_size", bs),
                ("Max New Tokens", "max_new_tokens", max_new_tokens),
                ("Top P", "top_p", top_p),
                ("Top K", "top_k", top_k),
                ("Temperature", "temperature", temperature),
                ("Repetition Penalty", "repetition_penalty", repetition_penalty),
                ("Elapsed(s)", "elapsed_s", f"{elapsed_s:.3f}"),
            ],
        )

    @synchronized_execution
    def dialogue(
        self,
        script,
        role_1_name,
        role_1_audio,
        role_1_ref_text,
        role_2_name,
        role_2_audio,
        role_2_ref_text,
        role_3_name,
        role_3_audio,
        role_3_ref_text,
        role_4_name,
        role_4_audio,
        role_4_ref_text,
        model_choice,
        user_did=None,
        device="auto",
        precision="bf16",
        language="Auto",
        pause_linebreak=0.5,
        period_pause=0.4,
        comma_pause=0.2,
        question_pause=0.6,
        hyphen_pause=0.3,
        merge_outputs=True,
        batch_size=4,
        seed=0,
        max_new_tokens_per_line=4096,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
        progress_callback=None,
    ):
        t0 = time.perf_counter()
        try:
            bs = int(batch_size)
        except Exception:
            bs = 1
        if bs < 1:
            bs = 1
        if bs > 16:
            bs = 16
        prompt_node = VoiceClonePromptNode()
        role_bank_node = RoleBankNode()
        dialogue_node = DialogueInferenceNode()

        prompts = []
        names = []
        if callable(progress_callback):
            try:
                progress_callback(0, "准备角色音色")
            except Exception:
                pass
        role_items = [
            (role_1_name, role_1_audio, role_1_ref_text),
            (role_2_name, role_2_audio, role_2_ref_text),
            (role_3_name, role_3_audio, role_3_ref_text),
            (role_4_name, role_4_audio, role_4_ref_text),
        ]
        roles_with_audio = [(n, a, t) for (n, a, t) in role_items if n and a is not None]
        for role_name, role_audio, role_ref_text in [
            (role_1_name, role_1_audio, role_1_ref_text),
            (role_2_name, role_2_audio, role_2_ref_text),
            (role_3_name, role_3_audio, role_3_ref_text),
            (role_4_name, role_4_audio, role_4_ref_text),
        ]:
            if role_name and role_audio is not None:
                audio_dict = self._audio_input_to_comfy_audio(role_audio)
                prompt = prompt_node.create_prompt(
                    ref_audio=audio_dict,
                    ref_text=role_ref_text or "",
                    model_choice=model_choice,
                    device=device,
                    precision=precision,
                    attention=attention,
                    x_vector_only=False,
                    unload_model_after_generate=unload_model_after_generate,
                )[0]
                prompts.append(prompt)
                names.append(role_name)
                if callable(progress_callback):
                    try:
                        pct = int(len(prompts) * 30 / max(len(roles_with_audio), 1))
                        progress_callback(pct, f"准备角色音色：{len(prompts)}/{len(roles_with_audio)}")
                    except Exception:
                        pass

        kwargs = {}
        for i, (name, prompt) in enumerate(zip(names, prompts), start=1):
            kwargs[f"role_name_{i}"] = name
            kwargs[f"prompt_{i}"] = prompt
        role_bank = role_bank_node.create_bank(**kwargs)[0]

        if callable(progress_callback):
            try:
                progress_callback(40, "生成对话音频")
            except Exception:
                pass
        parts = self._parse_pause_markup(script)
        saw_pause_markup = any(k == "pause" for (k, _) in parts)
        if not saw_pause_markup:
            result = dialogue_node.generate_dialogue(
                script=script,
                role_bank=role_bank,
                model_choice=model_choice,
                device=device,
                precision=precision,
                language=language,
                pause_linebreak=pause_linebreak,
                period_pause=period_pause,
                comma_pause=comma_pause,
                question_pause=question_pause,
                hyphen_pause=hyphen_pause,
                merge_outputs=merge_outputs,
                batch_size=bs,
                seed=seed,
                max_new_tokens_per_line=max_new_tokens_per_line,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                attention=attention,
                unload_model_after_generate=unload_model_after_generate,
            )
        else:
            leading_pause_s = 0.0
            idx = 0
            while idx < len(parts) and parts[idx][0] == "pause":
                try:
                    leading_pause_s += float(parts[idx][1])
                except Exception:
                    leading_pause_s += 0.0
                idx += 1
            if leading_pause_s > 120.0:
                leading_pause_s = 120.0
            audios: List[Dict[str, Any]] = []
            sr_known = None
            cursor = idx
            while cursor < len(parts):
                kind, payload = parts[cursor]
                if kind == "pause":
                    cursor += 1
                    continue
                seg_script = "" if payload is None else str(payload)
                if not seg_script.strip():
                    cursor += 1
                    continue
                if callable(progress_callback):
                    try:
                        progress_callback(40, "生成对话音频（分段）")
                    except Exception:
                        pass
                seg_result = dialogue_node.generate_dialogue(
                    script=seg_script,
                    role_bank=role_bank,
                    model_choice=model_choice,
                    device=device,
                    precision=precision,
                    language=language,
                    pause_linebreak=pause_linebreak,
                    period_pause=period_pause,
                    comma_pause=comma_pause,
                    question_pause=question_pause,
                    hyphen_pause=hyphen_pause,
                    merge_outputs=True,
                    batch_size=bs,
                    seed=seed,
                    max_new_tokens_per_line=max_new_tokens_per_line,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    attention=attention,
                    unload_model_after_generate=False,
                )
                seg_audio = self._extract_audio_dict(seg_result)
                sr_known = int(seg_audio.get("sample_rate", 24000))
                if not audios and leading_pause_s > 0.0:
                    audios.append(self._make_silence_audio_dict(int(sr_known), leading_pause_s, seg_audio))
                    leading_pause_s = 0.0
                audios.append(seg_audio)
                cursor += 1
                pause_total = 0.0
                while cursor < len(parts) and parts[cursor][0] == "pause":
                    try:
                        pause_total += float(parts[cursor][1])
                    except Exception:
                        pause_total += 0.0
                    cursor += 1
                if pause_total > 0.0 and sr_known is not None:
                    audios.append(self._make_silence_audio_dict(int(sr_known), float(min(pause_total, 120.0)), seg_audio))
            merged_dialogue = self._merge_audio_dicts(audios, gap_seconds=0.0) if len(audios) > 1 else audios[0]
            result = (merged_dialogue,)
            if unload_model_after_generate:
                try:
                    unload_qwen_tts_models()
                except Exception:
                    pass
        if callable(progress_callback):
            try:
                progress_callback(90, "合并输出")
            except Exception:
                pass
        sr, wav = self._process_output(result)
        elapsed_s = time.perf_counter() - t0
        logging.info("QwenTTS dialogue done: batch_size=%s, elapsed_s=%.3f", bs, elapsed_s)
        if callable(progress_callback):
            try:
                progress_callback(100, f"完成，批次大小: {bs}，耗时: {elapsed_s:.3f}秒")
            except Exception:
                pass
        return self._save_wav(
            sr,
            wav,
            "tts_dialogue",
            user_did,
            log_metadata=[
                ("Mode", "mode", "dialogue"),
                ("Model", "model_choice", model_choice),
                ("Language", "language", language),
                ("Seed", "seed", seed),
                ("Role1", "role_1_name", role_1_name),
                ("Role2", "role_2_name", role_2_name),
                ("Role3", "role_3_name", role_3_name),
                ("Role4", "role_4_name", role_4_name),
                ("Script", "script", script),
                ("Pause Linebreak", "pause_linebreak", pause_linebreak),
                ("Period Pause", "period_pause", period_pause),
                ("Comma Pause", "comma_pause", comma_pause),
                ("Question Pause", "question_pause", question_pause),
                ("Hyphen Pause", "hyphen_pause", hyphen_pause),
                ("Batch Size", "batch_size", bs),
                ("Max New Tokens/Line", "max_new_tokens_per_line", max_new_tokens_per_line),
                ("Top P", "top_p", top_p),
                ("Top K", "top_k", top_k),
                ("Temperature", "temperature", temperature),
                ("Repetition Penalty", "repetition_penalty", repetition_penalty),
                ("Elapsed(s)", "elapsed_s", f"{elapsed_s:.3f}"),
            ],
        )

    def _audio_input_to_comfy_audio(self, audio):
        if audio is None:
            raise ValueError("Missing reference audio")
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            return audio
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            sr, wav = audio
            if wav is None:
                raise ValueError("Missing reference audio waveform")
            if hasattr(wav, "cpu"):
                wav = wav.cpu().numpy()
            wav = np.asarray(wav)
            if np.issubdtype(wav.dtype, np.integer):
                info = np.iinfo(wav.dtype)
                denom = float(max(abs(info.min), abs(info.max)))
                wav = wav.astype(np.float32, copy=False) / denom
            else:
                wav = wav.astype(np.float32, copy=False)
                peak = float(np.max(np.abs(wav))) if wav.size else 0.0
                if peak > 1.5:
                    if peak <= 40000.0:
                        wav = wav / 32768.0
                    else:
                        wav = wav / peak
            wav = np.clip(wav, -1.0, 1.0)
            return {"waveform": wav, "sample_rate": int(sr)}
        raise ValueError("Unsupported reference audio format")

    def _process_output(self, result):
        """Convert ComfyUI audio output dict to (sr, waveform) for Gradio Audio"""
        # ComfyUI audio format: {"waveform": tensor/np [1, T] or [1, C, T], "sample_rate": int}
        if not result or not isinstance(result, tuple):
            raise ValueError("Invalid output from node")
        
        audio_dict = result[0]
        if "waveform" in audio_dict and "sample_rate" in audio_dict:
            sr = audio_dict["sample_rate"]
            wav = audio_dict["waveform"]
            
            # Convert tensor to numpy if needed
            if hasattr(wav, "cpu"):
                wav = wav.squeeze().cpu().numpy()
            elif isinstance(wav, np.ndarray):
                wav = wav.squeeze()
                
            if isinstance(wav, np.ndarray) and np.issubdtype(wav.dtype, np.floating):
                wav = np.clip(wav.astype(np.float32, copy=False), -1.0, 1.0)
            return (sr, wav)
        raise ValueError("Missing waveform or sample_rate")

qwen_tts_handler = QwenTTSWrapper()
