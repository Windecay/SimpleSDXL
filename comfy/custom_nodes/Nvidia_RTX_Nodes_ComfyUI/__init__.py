from __future__ import annotations

import io as python_io
import itertools
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from enum import Enum
from fractions import Fraction
from typing import Optional, TypedDict

import av
import folder_paths
import nvvfx
import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io

logger = logging.getLogger(__name__)

class _LazyVideoFrameSequence:
    def __init__(self, source, frame_count: int):
        self._source = source
        self._frame_count = frame_count
        self._first_frame = None

    def __len__(self):
        return self._frame_count

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError("Only index 0 is supported for lazy image sequence.")
        if self._first_frame is not None:
            return self._first_frame
        if isinstance(self._source, python_io.BytesIO):
            self._source.seek(0)
        with av.open(self._source, mode="r") as container:
            if not len(container.streams.video):
                raise ValueError("Video does not contain a video stream.")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                image = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                self._first_frame = image
                return image
        raise IndexError("Video contains no frames.")

    def __iter__(self):
        first = self[0]
        yield first
        if isinstance(self._source, python_io.BytesIO):
            self._source.seek(0)
        with av.open(self._source, mode="r") as container:
            stream = container.streams.video[0]
            skipped_first = False
            for frame in container.decode(stream):
                if not skipped_first:
                    skipped_first = True
                    continue
                yield torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0


def _decode_audio_from_video_source(source) -> Optional[Input.Audio]:
    with av.open(source, mode="r") as container:
        if not len(container.streams.audio):
            return None
        audio_stream = container.streams.audio[-1]
        resample = av.audio.resampler.AudioResampler(format="fltp").resample
        frames = itertools.chain.from_iterable(map(resample, container.decode(audio_stream)))
        audio_frames = []
        for frame in frames:
            audio_frames.append(frame.to_ndarray())
        if len(audio_frames) == 0:
            return None
        audio_data = np.concatenate(audio_frames, axis=1)
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        return {
            "waveform": audio_tensor,
            "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
        }


class _LazyAudioFromVideoSource(dict):
    def __init__(self, source):
        super().__init__()
        self._source = source
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        data = _decode_audio_from_video_source(self._source)
        if data is not None:
            self.update(data)
        self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class CacheBackend(str, Enum):
    MEMORY = "memory"
    DISK = "disk"


class CompressionMode(str, Enum):
    NEAR_LOSSLESS = "near_lossless"
    BALANCED = "balanced"
    COMPACT = "compact"


class UpscaleTypedDict(TypedDict):
    resize_type: UpscaleType
    scale: float
    width: int
    height: int


QUALITY_MAPPING = {
    "LOW": nvvfx.effects.QualityLevel.LOW,
    "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
    "HIGH": nvvfx.effects.QualityLevel.HIGH,
    "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
}


def _resolve_output_dimensions(width: int, height: int, resize_type: UpscaleTypedDict) -> tuple[int, int]:
    selected_type = resize_type["resize_type"]
    if selected_type == UpscaleType.SCALE_BY:
        scale = resize_type["scale"]
        output_width = int(width * scale)
        output_height = int(height * scale)
    elif selected_type == UpscaleType.TARGET_DIMENSIONS:
        output_width = resize_type["width"]
        output_height = resize_type["height"]
    else:
        raise ValueError(f"Unsupported resize type: {selected_type}")
    output_width = max(8, round(output_width / 8) * 8)
    output_height = max(8, round(output_height / 8) * 8)
    return output_width, output_height


def _get_selected_quality(quality: str):
    return QUALITY_MAPPING.get(quality, nvvfx.effects.QualityLevel.HIGH)


def _get_frame_batch_size(output_width: int, output_height: int, max_pixels_per_batch: int) -> int:
    out_pixels = max(1, output_width * output_height)
    return max(1, max_pixels_per_batch // out_pixels)


def _get_cache_dir() -> str:
    cache_dir = os.path.join(folder_paths.get_input_directory(), "nvidia_rtx_nodes_cache")
    os.makedirs(cache_dir, exist_ok=True)
    expire_before = time.time() - 24 * 60 * 60
    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getmtime(path) < expire_before:
                os.remove(path)
        except OSError:
            pass
    return cache_dir


def _create_temp_mp4_path(prefix: str) -> str:
    return os.path.join(_get_cache_dir(), f"{prefix}_{uuid.uuid4().hex}.mp4")


def _estimate_video_bitrate(width: int, height: int, fps: float, compression: str) -> int:
    bits_per_pixel = {
        CompressionMode.NEAR_LOSSLESS.value: 0.8,
        CompressionMode.BALANCED.value: 0.35,
        CompressionMode.COMPACT.value: 0.18,
    }.get(compression, 0.8)
    bitrate = int(width * height * max(fps, 1.0) * bits_per_pixel)
    return max(4_000_000, min(200_000_000, bitrate))


def _mux_packets(container: av.container.OutputContainer, packets) -> None:
    if packets is None:
        return
    if isinstance(packets, list):
        for packet in packets:
            container.mux(packet)
        return
    container.mux(packets)


def _upscale_batch(sr: nvvfx.VideoSuperRes, batch: torch.Tensor) -> torch.Tensor:
    batch_cuda = batch.cuda().permute(0, 3, 1, 2).float().contiguous()
    out_tensor = torch.empty(
        (batch.shape[0], sr.output_height, sr.output_width, batch.shape[-1]),
        device=batch.device,
        dtype=batch.dtype,
    )
    for index in range(batch_cuda.shape[0]):
        dlpack_out = sr.run(batch_cuda[index]).image
        out_tensor[index:index + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)
    return out_tensor


def _upscale_batch_streaming(sr: nvvfx.VideoSuperRes, batch: torch.Tensor) -> torch.Tensor:
    batch_cuda = batch.cuda().permute(0, 3, 1, 2).contiguous()
    if batch_cuda.dtype == torch.uint8:
        batch_cuda = batch_cuda.float().div_(255.0)
    else:
        batch_cuda = batch_cuda.float()
    out_tensor = torch.empty(
        (batch.shape[0], sr.output_height, sr.output_width, batch.shape[-1]),
        device=batch.device,
        dtype=torch.float32,
    )
    for index in range(batch_cuda.shape[0]):
        dlpack_out = sr.run(batch_cuda[index]).image
        out_tensor[index:index + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)
    return out_tensor


def _encode_frames(
    container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    frames: torch.Tensor,
) -> None:
    if frames is None:
        return
    if isinstance(frames, torch.Tensor):
        if frames.dtype == torch.uint8:
            batch_u8 = frames[..., :3]
        else:
            batch_u8 = (frames[..., :3] * 255.0).clamp(0, 255).to(torch.uint8)
        batch_np = batch_u8.cpu().numpy()
        for image in batch_np:
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            _mux_packets(container, stream.encode(video_frame))
        return
    for frame in frames:
        image = (frame[..., :3] * 255).clamp(0, 255).byte().cpu().numpy()
        video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        _mux_packets(container, stream.encode(video_frame))


def _process_sequence_chunk(
    sr: nvvfx.VideoSuperRes,
    container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    images: torch.Tensor,
    frame_batch_size: int,
) -> int:
    frame_count = 0
    t_upscale = 0.0
    t_encode = 0.0
    for start in range(0, images.shape[0], frame_batch_size):
        batch = images[start:start + frame_batch_size]
        t0 = time.perf_counter()
        upscaled_batch = _upscale_batch_streaming(sr, batch)
        t1 = time.perf_counter()
        _encode_frames(container, stream, upscaled_batch)
        t2 = time.perf_counter()
        frame_count += upscaled_batch.shape[0]
        t_upscale += (t1 - t0)
        t_encode += (t2 - t1)

    return frame_count


def _iter_video_frame_chunks(video_path: str, chunk_frames: int):
    with av.open(video_path, mode="r") as container:
        if not len(container.streams.video):
            raise ValueError("Input video does not contain a video stream.")
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= chunk_frames:
                yield torch.from_numpy(np.stack(frames, axis=0))
                frames = []
        if frames:
            yield torch.from_numpy(np.stack(frames, axis=0))


def _add_video_stream(
    output_container: av.container.OutputContainer,
    preferred_encoder: str,
    frame_rate: Fraction,
):
    if preferred_encoder == "auto":
        encoder_candidates = ["h264_nvenc", "hevc_nvenc", "h264"]
    else:
        encoder_candidates = [preferred_encoder, "h264"]

    errors = []
    for codec_name in encoder_candidates:
        try:
            stream = output_container.add_stream(codec_name, rate=frame_rate)
            fallback_reason = None
            if len(errors) > 0:
                fallback_reason = " | ".join(errors)
            return stream, codec_name, fallback_reason
        except Exception as e:
            errors.append(f"{codec_name}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Failed to create video encoder stream. " + " | ".join(errors))


def _get_ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _start_ffmpeg_rawvideo_writer(
    output_path: str,
    width: int,
    height: int,
    fps: Fraction,
    bitrate: int,
    encoder: str,
):
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg is None:
        return None, "ffmpeg not found"

    max_dim = max(width, height)
    if encoder == "auto":
        encoder_candidates = ["h264_nvenc"] if max_dim <= 4096 else ["hevc_nvenc", "h264_nvenc"]
    elif encoder == "h264_nvenc":
        encoder_candidates = ["h264_nvenc"] if max_dim <= 4096 else ["hevc_nvenc", "h264_nvenc"]
    elif encoder == "hevc_nvenc":
        encoder_candidates = ["hevc_nvenc"]
    elif encoder == "hevc":
        encoder_candidates = ["libx265", "libx264"]
    else:
        encoder_candidates = ["libx264"]

    errors = []
    for vcodec in encoder_candidates:
        args = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            vcodec,
            "-b:v",
            str(int(bitrate)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return proc, vcodec
        except Exception as e:
            errors.append(f"{vcodec}: {type(e).__name__}: {e}")
            continue
    return None, " | ".join(errors) if errors else "unknown error"


def _write_rawvideo_frames(proc: subprocess.Popen, frames: torch.Tensor) -> None:
    if proc.stdin is None:
        raise RuntimeError("ffmpeg process stdin is not available")
    if proc.poll() is not None:
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        raise RuntimeError(f"ffmpeg exited early with code {proc.returncode}: {stderr.decode(errors='replace')}")
    if frames.dtype == torch.uint8:
        batch_u8 = frames[..., :3].contiguous()
    else:
        batch_u8 = (frames[..., :3] * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
    try:
        proc.stdin.write(batch_u8.cpu().numpy().tobytes())
    except BrokenPipeError:
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        raise RuntimeError(f"ffmpeg broken pipe: {stderr.decode(errors='replace')}")


def _ffmpeg_mux_audio_from_source(video_path: str, audio_source_path: str) -> bool:
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg is None:
        return False
    tmp_out = f"{video_path}.audio_tmp_{uuid.uuid4().hex}.mp4"
    args = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        tmp_out,
    ]
    try:
        res = subprocess.run(args, capture_output=True)
        if res.returncode != 0:
            return False
        os.replace(tmp_out, video_path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except OSError:
            pass
        return False

def _write_audio_from_input(
    container: av.container.OutputContainer,
    audio_stream: av.audio.stream.AudioStream,
    audio: Input.Audio,
    frame_rate: Fraction,
    frame_count: int,
) -> bool:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.shape[0] == 0 or waveform.shape[1] == 0:
        return False
    trimmed_waveform = waveform[0, :, :math.ceil((sample_rate / frame_rate) * frame_count)]
    if trimmed_waveform.shape[-1] == 0:
        return False
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(trimmed_waveform.shape[0], "stereo")
    audio_frame = av.AudioFrame.from_ndarray(trimmed_waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
    audio_frame.sample_rate = sample_rate
    audio_frame.pts = 0
    _mux_packets(container, audio_stream.encode(audio_frame))
    _mux_packets(container, audio_stream.encode(None))
    return True


def _open_output_target(cache_backend: str):
    if cache_backend == CacheBackend.MEMORY.value:
        return python_io.BytesIO(), None, CacheBackend.MEMORY.value, None
    try:
        output_path = _create_temp_mp4_path("rtx_vsr_output")
        with open(output_path, "wb"):
            pass
        with av.open(output_path, mode="w", format="mp4"):
            pass
        return output_path, output_path, CacheBackend.DISK.value, None
    except Exception as e:
        return python_io.BytesIO(), None, CacheBackend.MEMORY.value, f"{type(e).__name__}: {e}"


def _run_chunked_upscale(
    resize_type: UpscaleTypedDict,
    quality: str,
    cache_backend: str,
    chunk_frames: int,
    max_pixels_per_batch: int,
    compression: str,
    fps: float,
    encoder: str,
    writer: str,
    video: Optional[Input.Video] = None,
    images: Optional[torch.Tensor] = None,
    audio: Optional[Input.Audio] = None,
) -> tuple[Input.Video, str, object, int, float, str]:
    start_ts = time.perf_counter()
    if video is None and images is None:
        raise ValueError("Connect either video or images.")
    if video is not None and images is not None:
        raise ValueError("Connect only one source: video or images.")

    normalized_input_path = None
    audio_mode = "none"
    source_kind = "video" if video is not None else "images"

    if video is not None:
        source_width, source_height = video.get_dimensions()
        source_fps = float(video.get_frame_rate())
        normalized_input_path = _create_temp_mp4_path("rtx_vsr_input")
        video.save_to(
            normalized_input_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
        )
        if audio is not None:
            audio_mode = "input"
        else:
            audio_mode = "source"
    else:
        if images.shape[0] == 0:
            raise ValueError("Images input is empty.")
        _, source_height, source_width, _ = images.shape
        source_fps = fps if fps and fps > 0 else 30.0
        if audio is not None:
            audio_mode = "input"

    output_width, output_height = _resolve_output_dimensions(source_width, source_height, resize_type)
    frame_batch_size = _get_frame_batch_size(output_width, output_height, max_pixels_per_batch)
    output_target, output_path, backend_used, backend_fallback_reason = _open_output_target(cache_backend)
    bitrate = _estimate_video_bitrate(output_width, output_height, source_fps, compression)
    frame_rate = Fraction(round(source_fps * 1000), 1000)
    total_frames = 0
    total_chunks = 0
    expected_chunks = None
    source_audio_container = None
    source_audio_stream = None
    output_audio_stream = None

    if images is not None:
        expected_chunks = int(math.ceil(images.shape[0] / max(1, chunk_frames)))
    elif normalized_input_path is not None:
        try:
            expected_frame_count = InputImpl.VideoFromFile(normalized_input_path).get_frame_count()
            expected_chunks = int(math.ceil(expected_frame_count / max(1, chunk_frames)))
        except Exception:
            expected_chunks = None

    logger.info(
        "RTX VSR chunked start: source=%s %sx%s fps=%.3f -> out=%sx%s; chunk_frames=%d; max_pixels_per_batch=%d; cache_backend=%s; compression=%s; quality=%s",
        source_kind,
        source_width,
        source_height,
        source_fps,
        output_width,
        output_height,
        chunk_frames,
        max_pixels_per_batch,
        cache_backend,
        compression,
        quality,
    )
    if backend_used == CacheBackend.DISK.value and output_path is not None:
        logger.info("RTX VSR output cache file: %s", output_path)
    if backend_used != cache_backend:
        logger.warning(
            "RTX VSR cache backend fallback: requested=%s used=%s reason=%s",
            cache_backend,
            backend_used,
            backend_fallback_reason,
        )

    try:
        open_kwargs = {"mode": "w", "options": {"movflags": "use_metadata_tags"}}
        if isinstance(output_target, python_io.BytesIO):
            open_kwargs["format"] = "mp4"

        if writer in ("auto", "ffmpeg"):
            logger.info("RTX VSR ffmpeg path: %s", _get_ffmpeg_path())

        can_use_ffmpeg = (
            writer in ("auto", "ffmpeg")
            and output_path is not None
            and not isinstance(output_target, python_io.BytesIO)
            and audio is None
        )
        if writer in ("auto", "ffmpeg") and not can_use_ffmpeg:
            reasons = []
            if output_path is None:
                reasons.append("no_output_path")
            if isinstance(output_target, python_io.BytesIO):
                reasons.append("memory_backend")
            if audio is not None:
                reasons.append("audio_input_connected")
            logger.info("RTX VSR ffmpeg writer disabled: %s", ",".join(reasons) if reasons else "unknown")

        ffmpeg_proc = None
        encoder_used = None
        if can_use_ffmpeg:
            ffmpeg_proc, encoder_used = _start_ffmpeg_rawvideo_writer(
                output_path=output_path,
                width=output_width,
                height=output_height,
                fps=frame_rate,
                bitrate=bitrate,
                encoder=encoder,
            )
            if ffmpeg_proc is None:
                logger.warning("RTX VSR ffmpeg writer unavailable, falling back to PyAV: %s", encoder_used)
                encoder_used = None
        if ffmpeg_proc is not None:
            logger.info("RTX VSR writer: ffmpeg (%s)", encoder_used)
            with nvvfx.VideoSuperRes(_get_selected_quality(quality)) as sr:
                sr.output_width = output_width
                sr.output_height = output_height
                sr.load()
                if normalized_input_path is not None:
                    chunk_iter = _iter_video_frame_chunks(normalized_input_path, chunk_frames)
                else:
                    chunk_iter = (images[start:start + chunk_frames] for start in range(0, images.shape[0], chunk_frames))

                for chunk_index, chunk in enumerate(chunk_iter, start=1):
                    chunk_start_ts = time.perf_counter()
                    t_upscale = 0.0
                    t_encode = 0.0
                    for bs in range(0, chunk.shape[0], frame_batch_size):
                        batch = chunk[bs:bs + frame_batch_size]
                        t0 = time.perf_counter()
                        upscaled_batch = _upscale_batch_streaming(sr, batch)
                        t1 = time.perf_counter()
                        _write_rawvideo_frames(ffmpeg_proc, upscaled_batch)
                        t2 = time.perf_counter()
                        total_frames += upscaled_batch.shape[0]
                        t_upscale += (t1 - t0)
                        t_encode += (t2 - t1)
                    total_chunks = chunk_index
                    logger.info(
                        "RTX VSR chunk %d/%s processed: frames=%d elapsed=%.2fs total_frames=%d",
                        chunk_index,
                        expected_chunks if expected_chunks is not None else "?",
                        chunk.shape[0],
                        time.perf_counter() - chunk_start_ts,
                        total_frames,
                    )
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            stderr = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr is not None else b""
            ret = ffmpeg_proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg failed with code {ret}: {stderr.decode(errors='replace')}")
            if normalized_input_path is not None:
                if _ffmpeg_mux_audio_from_source(output_path, normalized_input_path):
                    audio_mode = "source"
                else:
                    audio_mode = "none"
        else:
            with av.open(output_target, **open_kwargs) as output_container:
                video_stream, encoder_used, encoder_fallback_reason = _add_video_stream(output_container, encoder, frame_rate)
                video_stream.width = output_width
                video_stream.height = output_height
                video_stream.pix_fmt = "yuv420p"
                video_stream.bit_rate = bitrate
                if encoder_fallback_reason is not None:
                    logger.warning("RTX VSR encoder fallback: requested=%s used=%s reason=%s", encoder, encoder_used, encoder_fallback_reason)
                elif encoder_used != encoder and encoder != "auto":
                    logger.warning("RTX VSR encoder fallback: requested=%s used=%s", encoder, encoder_used)
                else:
                    logger.info("RTX VSR encoder: %s", encoder_used)

                if audio is not None:
                    waveform = audio["waveform"]
                    sample_rate = int(audio["sample_rate"])
                    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[1], "stereo")
                    output_audio_stream = output_container.add_stream("aac", rate=sample_rate, layout=layout)
                elif normalized_input_path is not None:
                    source_audio_container = av.open(normalized_input_path, mode="r")
                    if len(source_audio_container.streams.audio):
                        source_audio_stream = source_audio_container.streams.audio[-1]
                        output_audio_stream = output_container.add_stream_from_template(template=source_audio_stream, opaque=True)
                    else:
                        source_audio_container.close()
                        source_audio_container = None

                with nvvfx.VideoSuperRes(_get_selected_quality(quality)) as sr:
                    sr.output_width = output_width
                    sr.output_height = output_height
                    sr.load()

                    if normalized_input_path is not None:
                        for chunk_index, chunk in enumerate(_iter_video_frame_chunks(normalized_input_path, chunk_frames), start=1):
                            chunk_start_ts = time.perf_counter()
                            processed = _process_sequence_chunk(sr, output_container, video_stream, chunk, frame_batch_size)
                            total_frames += processed
                            total_chunks = chunk_index
                            logger.info(
                                "RTX VSR chunk %d/%s processed: frames=%d elapsed=%.2fs total_frames=%d",
                                chunk_index,
                                expected_chunks if expected_chunks is not None else "?",
                                processed,
                                time.perf_counter() - chunk_start_ts,
                                total_frames,
                            )
                    else:
                        for chunk_index, start in enumerate(range(0, images.shape[0], chunk_frames), start=1):
                            chunk_start_ts = time.perf_counter()
                            processed = _process_sequence_chunk(
                                sr,
                                output_container,
                                video_stream,
                                images[start:start + chunk_frames],
                                frame_batch_size,
                            )
                            total_frames += processed
                            total_chunks = chunk_index
                            logger.info(
                                "RTX VSR chunk %d/%s processed: frames=%d elapsed=%.2fs total_frames=%d",
                                chunk_index,
                                expected_chunks if expected_chunks is not None else "?",
                                processed,
                                time.perf_counter() - chunk_start_ts,
                                total_frames,
                            )

                if total_frames == 0:
                    raise ValueError("No frames were processed.")

                _mux_packets(output_container, video_stream.encode(None))

                if audio is not None:
                    _write_audio_from_input(output_container, output_audio_stream, audio, frame_rate, total_frames)
                elif source_audio_container is not None and source_audio_stream is not None and output_audio_stream is not None:
                    for packet in source_audio_container.demux(source_audio_stream):
                        if packet.dts is None:
                            continue
                        packet.stream = output_audio_stream
                        output_container.mux(packet)
                elif normalized_input_path is not None:
                    audio_mode = "none"

        output_source = output_target
        if isinstance(output_target, python_io.BytesIO):
            output_target.seek(0)
            result_video = InputImpl.VideoFromFile(output_target)
        else:
            result_video = InputImpl.VideoFromFile(output_path)

        cache_info = (
            f"backend={backend_used}(requested={cache_backend}); frames={total_frames}; "
            f"size={output_width}x{output_height}; fps={float(frame_rate):.3f}; "
            f"audio={audio_mode}; compression={compression}; encoder={encoder_used}"
        )
        elapsed = time.perf_counter() - start_ts
        logger.info(
            "RTX VSR chunked done: chunks=%d/%s frames=%d elapsed=%.2fs fps=%.2f cache_backend=%s",
            total_chunks,
            expected_chunks if expected_chunks is not None else "?",
            total_frames,
            elapsed,
            (total_frames / elapsed) if elapsed > 0 else 0.0,
            backend_used,
        )
        return result_video, cache_info, output_source, total_frames, source_fps, backend_used
    finally:
        if source_audio_container is not None:
            source_audio_container.close()
        if normalized_input_path and os.path.exists(normalized_input_path):
            os.remove(normalized_input_path)


class RTXVideoSuperResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels.")
                        ])
                    ],
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, height, width, _ = images.shape
        output_width, output_height = _resolve_output_dimensions(width, height, resize_type)
        batch_size = _get_frame_batch_size(output_width, output_height, 1024 * 1024 * 16)
        selected_quality = _get_selected_quality(quality)
        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()
            out_tensor = torch.empty(
                (images.shape[0], output_height, output_width, images.shape[-1]),
                device=images.device,
                dtype=images.dtype,
            )
            for start in range(0, images.shape[0], batch_size):
                out_tensor[start:start + batch_size] = _upscale_batch(sr, images[start:start + batch_size])
        return io.NodeOutput(out_tensor)


class RTXVideoSuperResolutionChunked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolutionChunked",
            display_name="RTX Video Super Resolution Chunked",
            category="image/video",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr", "chunked", "streaming", "long video"],
            inputs=[
                io.Video.Input("video", optional=True, tooltip="Preferred for long videos. The node will stream frames in segments and emit a compressed VIDEO output."),
                io.Image.Input("images", optional=True, tooltip="Optional image sequence input. Use video input when you want the lowest memory usage."),
                io.Audio.Input("audio", optional=True, tooltip="Optional audio track. When video input is used and this is left empty, the source audio is copied when possible."),
                io.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.01, tooltip="Used only when images input is connected. 0 means auto (defaults to 30).", advanced=True),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels."),
                        ]),
                    ],
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
                io.Combo.Input(
                    "cache_backend",
                    options=CacheBackend,
                    default=CacheBackend.DISK,
                    tooltip="Store the compressed intermediate result in memory or on disk before passing it to the next node.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=120,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Frames decoded and processed per segment. Lower values use less memory, higher values improve throughput.",
                ),
                io.Int.Input(
                    "max_pixels_per_batch",
                    default=1024 * 1024 * 16,
                    min=1024 * 1024,
                    max=1024 * 1024 * 256,
                    step=1024 * 1024,
                    tooltip="Upper bound for total output pixels processed on the GPU at once.",
                ),
                io.Combo.Input(
                    "encoder",
                    options=["auto", "h264", "h264_nvenc", "hevc", "hevc_nvenc"],
                    default="auto",
                    tooltip="Video encoder. auto tries h264_nvenc first then falls back to h264.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "writer",
                    options=["auto", "pyav", "ffmpeg"],
                    default="auto",
                    tooltip="Video writer backend. auto uses ffmpeg when possible, otherwise uses PyAV.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "compression",
                    options=CompressionMode,
                    default=CompressionMode.NEAR_LOSSLESS,
                    tooltip="Controls the bitrate of the compressed VIDEO output.",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="upscaled_video"),
                io.String.Output("cache_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        resize_type: UpscaleTypedDict,
        quality: str,
        cache_backend: str,
        chunk_frames: int,
        max_pixels_per_batch: int,
        encoder: str,
        writer: str,
        compression: str,
        fps: float,
        video: Optional[Input.Video] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Input.Audio] = None,
    ) -> io.NodeOutput:
        result_video, cache_info, _, _, _, _ = _run_chunked_upscale(
            resize_type=resize_type,
            quality=quality,
            cache_backend=cache_backend,
            chunk_frames=chunk_frames,
            max_pixels_per_batch=max_pixels_per_batch,
            compression=compression,
            fps=fps,
            encoder=encoder,
            writer=writer,
            video=video,
            images=images,
            audio=audio,
        )
        return io.NodeOutput(result_video, cache_info)


class RTXVideoSuperResolutionChunkedImageSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolutionChunkedImageSequence",
            display_name="RTX Video Super Resolution Chunked Image Sequence",
            category="image/video",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr", "chunked", "image sequence", "legacy"],
            inputs=[
                io.Video.Input("video", optional=True, tooltip="Preferred for long videos. The node still processes in segments before exposing IMAGE output."),
                io.Image.Input("images", optional=True, tooltip="Optional image sequence input."),
                io.Audio.Input("audio", optional=True, tooltip="Optional audio input, used only while building the intermediate compressed video."),
                io.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.01, tooltip="Used only when images input is connected. 0 means auto (defaults to 30).", advanced=True),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels."),
                        ]),
                    ],
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
                io.Combo.Input(
                    "cache_backend",
                    options=CacheBackend,
                    default=CacheBackend.DISK,
                    tooltip="Intermediate chunked result is stored in memory or on disk before being expanded back to IMAGE output.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=120,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Frames decoded and processed per segment. Lower values use less memory during upscaling.",
                ),
                io.Int.Input(
                    "max_pixels_per_batch",
                    default=1024 * 1024 * 16,
                    min=1024 * 1024,
                    max=1024 * 1024 * 256,
                    step=1024 * 1024,
                    tooltip="Upper bound for total output pixels processed on the GPU at once.",
                ),
                io.Combo.Input(
                    "encoder",
                    options=["auto", "h264", "h264_nvenc", "hevc", "hevc_nvenc"],
                    default="auto",
                    tooltip="Intermediate encoder. auto tries h264_nvenc first then falls back to h264.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "writer",
                    options=["auto", "pyav", "ffmpeg"],
                    default="auto",
                    tooltip="Intermediate writer backend. auto uses ffmpeg when possible, otherwise uses PyAV.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "compression",
                    options=CompressionMode,
                    default=CompressionMode.NEAR_LOSSLESS,
                    tooltip="Controls the quality of the intermediate compressed video before it is expanded back to IMAGE output.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="upscaled_images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.String.Output("cache_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        resize_type: UpscaleTypedDict,
        quality: str,
        cache_backend: str,
        chunk_frames: int,
        max_pixels_per_batch: int,
        encoder: str,
        writer: str,
        compression: str,
        fps: float,
        video: Optional[Input.Video] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Input.Audio] = None,
    ) -> io.NodeOutput:
        _, cache_info, output_source, total_frames, source_fps, _ = _run_chunked_upscale(
            resize_type=resize_type,
            quality=quality,
            cache_backend=cache_backend,
            chunk_frames=chunk_frames,
            max_pixels_per_batch=max_pixels_per_batch,
            compression=compression,
            fps=fps,
            encoder=encoder,
            writer=writer,
            video=video,
            images=images,
            audio=None,
        )
        output_images = _LazyVideoFrameSequence(output_source, total_frames)
        if audio is not None:
            output_audio = audio
        elif video is not None:
            output_audio = _LazyAudioFromVideoSource(video.get_stream_source())
        else:
            output_audio = None
        output_fps = source_fps if video is not None else fps
        return io.NodeOutput(output_images, output_audio, output_fps, cache_info)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXVideoSuperResolution,
            RTXVideoSuperResolutionChunked,
            RTXVideoSuperResolutionChunkedImageSequence,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()

if False:
    NODE_CLASS_MAPPINGS = {
        "RTXVideoSuperResolution": RTXVideoSuperResolution,
        "RTXVideoSuperResolutionChunked": RTXVideoSuperResolutionChunked,
        "RTXVideoSuperResolutionChunkedImageSequence": RTXVideoSuperResolutionChunkedImageSequence,
    }
