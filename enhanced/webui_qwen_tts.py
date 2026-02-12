import os
import torch
import torchaudio
import numpy as np
import threading
import queue
from typing import Optional, List, Dict, Any, Tuple
import time
import datetime
import random
import wave

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

    def _save_wav(self, sr: int, wav: Any, prefix: str, user_did: Optional[str]) -> str:
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
        max_new_tokens=2048,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
    ):
        node = VoiceDesignNode()
        result = node.generate(
            text=text,
            instruct=instruct,
            model_choice=model_choice,
            device=device,
            precision=precision,
            language=language,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_model_after_generate=unload_model_after_generate,
        )
        sr, wav = self._process_output(result)
        return self._save_wav(sr, wav, "tts_voice_design", user_did)

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
        max_new_tokens=2048,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        x_vector_only=False,
        attention="auto",
        unload_model_after_generate=False,
        custom_model_path="",
    ):
        clone_node = VoiceCloneNode()
        audio_dict = self._audio_input_to_comfy_audio(ref_audio)
        result = clone_node.generate(
            target_text=target_text,
            model_choice=model_choice,
            device=device,
            precision=precision,
            language=language,
            ref_audio=audio_dict,
            ref_text=ref_text or "",
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            x_vector_only=x_vector_only,
            attention=attention,
            unload_model_after_generate=unload_model_after_generate,
            custom_model_path=custom_model_path or "",
        )
        sr, wav = self._process_output(result)
        return self._save_wav(sr, wav, "tts_voice_clone", user_did)

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
        max_new_tokens=2048,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
        custom_model_path="",
        custom_speaker_name="",
    ):
        node = CustomVoiceNode()
        result = node.generate(
            text=text,
            speaker=speaker,
            model_choice=model_choice,
            device=device,
            precision=precision,
            language=language,
            seed=seed,
            instruct=instruct or "",
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_model_after_generate=unload_model_after_generate,
            custom_model_path=custom_model_path or "",
            custom_speaker_name=custom_speaker_name or "",
        )
        sr, wav = self._process_output(result)
        return self._save_wav(sr, wav, "tts_custom_voice", user_did)

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
        max_new_tokens_per_line=2048,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        attention="auto",
        unload_model_after_generate=False,
    ):
        prompt_node = VoiceClonePromptNode()
        role_bank_node = RoleBankNode()
        dialogue_node = DialogueInferenceNode()

        prompts = []
        names = []
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

        kwargs = {}
        for i, (name, prompt) in enumerate(zip(names, prompts), start=1):
            kwargs[f"role_name_{i}"] = name
            kwargs[f"prompt_{i}"] = prompt
        role_bank = role_bank_node.create_bank(**kwargs)[0]

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
            batch_size=batch_size,
            seed=seed,
            max_new_tokens_per_line=max_new_tokens_per_line,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_model_after_generate=unload_model_after_generate,
        )
        sr, wav = self._process_output(result)
        return self._save_wav(sr, wav, "tts_dialogue", user_did)

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
                
            if isinstance(wav, np.ndarray) and wav.dtype == np.float32:
                wav = np.clip(wav, -1.0, 1.0)
                wav = (wav * 32767.0).astype(np.int16)
            return (sr, wav)
        raise ValueError("Missing waveform or sample_rate")

qwen_tts_handler = QwenTTSWrapper()
