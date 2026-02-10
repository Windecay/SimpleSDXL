import os
import subprocess
import numpy as np
import folder_paths
import tempfile
import soundfile as sf
from comfy.utils import ProgressBar
import json
from PIL import Image

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    ffmpeg_path = "ffmpeg"


class PainterVideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "frame_rate": ("FLOAT", {"default": 24, "min": 1, "max": 240, "step": 0.1, "display": "number"}),
                "format": (["video/h264-mp4", "video/webm", "image/gif"],),
                "filename_prefix": ("STRING", {"default": "Painter_Video"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    OUTPUT_NODE = True
    CATEGORY = "Painter/Video"
    FUNCTION = "combine_video"

    def combine_video(self, images, frame_rate, format, filename_prefix="Painter", audio=None, 
                      prompt=None, extra_pnginfo=None, unique_id=None):
        pbar = ProgressBar(len(images))
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        ext = "mp4" if "mp4" in format else ("webm" if "webm" in format else "gif")
        file_name = f"{filename}_{counter:05}_.{ext}"
        file_path = os.path.join(full_output_folder, file_name)

        # Build metadata
        video_metadata = {}
        if prompt is not None:
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                video_metadata[x] = extra_pnginfo[x]

        # Data validation
        images_np = images.cpu().numpy()
        if not np.isfinite(images_np).all():
            images_np = np.nan_to_num(images_np, nan=0.0, posinf=1.0, neginf=0.0)
        images_np = np.clip(images_np, 0, 1)
        images_np = (images_np * 255).astype(np.uint8)
        n, h, w, c = images_np.shape
        w, h = (w // 2) * 2, (h // 2) * 2

        # Build FFmpeg arguments
        args = [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(frame_rate), "-i", "-"
        ]

        audio_temp_path = None
        if audio is not None:
            try:
                wav_tensor = audio['waveform']
                wav_data = wav_tensor[0].cpu().numpy().transpose() if len(wav_tensor.shape) == 3 else wav_tensor.cpu().numpy().transpose()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    sf.write(temp_audio.name, wav_data, audio['sample_rate'], format='WAV')
                    audio_temp_path = temp_audio.name
                args += ["-i", audio_temp_path]
            except Exception as e:
                print(f"Warning: Audio processing failed: {e}")

        if ext == "mp4":
            args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "faster"]
            if audio_temp_path:
                args += ["-c:a", "aac", "-shortest"]
        elif ext == "webm":
            args += ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
            if audio_temp_path:
                args += ["-c:a", "libvorbis", "-shortest"]
        else:
            args += ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]

        # Add metadata if available
        metadata_path = None
        if video_metadata:
            try:
                metadata = json.dumps(video_metadata)
                metadata = metadata.replace("\\", "\\\\")
                metadata = metadata.replace(";", "\\;")
                metadata = metadata.replace("#", "\\#")
                metadata = metadata.replace("=", "\\=")
                metadata = metadata.replace("\n", "\\\n")
                metadata = "comment=" + metadata
                
                metadata_path = os.path.join(tempfile.gettempdir(), f"painter_metadata_{unique_id}.txt")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(";FFMETADATA1\n")
                    f.write(metadata)
                
                # Insert metadata input into ffmpeg args
                args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
            except Exception as e:
                print(f"Warning: Metadata processing failed: {e}")

        args.append(file_path)

        # Use temp file to avoid pipe deadlock
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.raw', delete=False) as temp_video:
            temp_video_path = temp_video.name
            
            for i, frame in enumerate(images_np):
                temp_video.write(frame[:h, :w, :].tobytes())
                pbar.update(1)
            
            temp_video.flush()
            os.fsync(temp_video.fileno())
        
        try:
            with open(temp_video_path, 'rb') as f:
                process = subprocess.Popen(
                    args,
                    stdin=f,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = process.communicate(timeout=60)
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"FFmpeg failed:\n{error_msg}")
            
            print(f"Video created!: {n} frames, {w}x{h}, {frame_rate}fps -> {file_name}")
            
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            if audio_temp_path and os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
                
            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)

        return {
            "ui": {"painter_output": [{"filename": file_name, "subfolder": subfolder, "type": "output"}]},
            "result": (file_name,)
        }


NODE_CLASS_MAPPINGS = {
    "PainterVideoCombine": PainterVideoCombine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterVideoCombine": "Painter Video Combine"
}
