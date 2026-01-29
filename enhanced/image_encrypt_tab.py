import functools
import math
from typing import Tuple

import gradio as gr
import numpy as np
from PIL import Image

import modules.gradio_hijack as grh


@functools.lru_cache(maxsize=16)
def _gilbert2d_indices(width: int, height: int) -> np.ndarray:
    total = int(width) * int(height)
    out = np.empty(total, dtype=np.int32)

    def _sign(x: int) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def _generate2d(
        x: int,
        y: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        pos: int,
    ) -> int:
        w = abs(ax + ay)
        h = abs(bx + by)

        dax, day = _sign(ax), _sign(ay)
        dbx, dby = _sign(bx), _sign(by)

        if h == 1:
            for _ in range(w):
                out[pos] = x + y * width
                pos += 1
                x += dax
                y += day
            return pos

        if w == 1:
            for _ in range(h):
                out[pos] = x + y * width
                pos += 1
                x += dbx
                y += dby
            return pos

        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2

        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if (w2 % 2) and (w > 2):
                ax2 += dax
                ay2 += day
            pos = _generate2d(x, y, ax2, ay2, bx, by, pos)
            pos = _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, pos)
            return pos

        if (h2 % 2) and (h > 2):
            bx2 += dbx
            by2 += dby

        pos = _generate2d(x, y, bx2, by2, ax2, ay2, pos)
        pos = _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, pos)
        pos = _generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
            pos,
        )
        return pos

    if width >= height:
        pos = _generate2d(0, 0, width, 0, 0, height, 0)
    else:
        pos = _generate2d(0, 0, 0, height, width, 0, 0)

    if pos != total:
        raise RuntimeError(f"gilbert2d fill mismatch: {pos} != {total} for {width}x{height}")

    return out


def _parse_password(password: str | None) -> Tuple[int, int, int]:
    step = 1
    v = 0
    h = 0
    if not password:
        return step, v, h

    pw = str(password)
    if len(pw) >= 2 and pw[0:2].isdigit():
        step = max(1, int(pw[0:2]))
    if len(pw) >= 3 and pw[2].isdigit():
        v = int(pw[2])
    if len(pw) >= 4 and pw[3].isdigit():
        h = int(pw[3])
    return step, v, h


def _add_padding(arr: np.ndarray, v: int, h: int) -> np.ndarray:
    if v <= 0 and h <= 0:
        return arr

    height, width = arr.shape[0], arr.shape[1]
    new_width = width + max(0, v)
    new_height = height + max(0, h)
    padded = np.empty((new_height, new_width, 4), dtype=arr.dtype)

    padded[:height, :width] = arr

    if v > 0:
        last_col = arr[:, width - 1 : width, :]
        padded[:height, width:] = np.repeat(last_col, v, axis=1)

    if h > 0:
        last_row = padded[height - 1 : height, :, :]
        padded[height:, :, :] = np.repeat(last_row, h, axis=0)

    return padded


def _obfuscate_pil(image: Image.Image | None, password: str | None, decrypt: bool) -> Image.Image | None:
    if image is None:
        return None

    step, v, h = _parse_password(password)
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)

    if decrypt and (v > 0 or h > 0):
        effective_width = max(1, arr.shape[1] - v)
        effective_height = max(1, arr.shape[0] - h)
        arr = arr[:effective_height, :effective_width, :]

    height, width = arr.shape[0], arr.shape[1]
    total = width * height
    if total <= 0:
        return None

    positions = _gilbert2d_indices(width, height)
    offset = int(round(((math.sqrt(5) - 1.0) / 2.0) * total)) % total
    new_positions = np.roll(positions, -offset)

    pixels = arr.reshape((total, 4)).copy()
    buffer = np.empty_like(pixels)
    if decrypt:
        for _ in range(step):
            buffer[positions] = pixels[new_positions]
            pixels, buffer = buffer, pixels
    else:
        for _ in range(step):
            buffer[new_positions] = pixels[positions]
            pixels, buffer = buffer, pixels

    out_arr = pixels.reshape((height, width, 4))
    if not decrypt and (v > 0 or h > 0):
        out_arr = _add_padding(out_arr, v, h)

    return Image.fromarray(out_arr, mode="RGBA")


def add_image_encrypt_tab(progress_window):
    with gr.Tab(label="Image Encrypt", id="image_encrypt_tab", visible=True):
        with gr.Column():
            input_image = grh.Image(label="Input Image", source="upload", type="pil")
            password = gr.Textbox(label="Password", value="", placeholder="0100")
            with gr.Row():
                encrypt_btn = gr.Button(value="Encrypt")
                decrypt_btn = gr.Button(value="Decrypt")
            gr.HTML('<div style="font-size: 12px; opacity: 0.75;">Source: https://dfqtphx.netlify.app/</div>')

        def _encrypt_to_progress(image: Image.Image, pw: str):
            result = _obfuscate_pil(image, pw, decrypt=False)
            return gr.update(value=result, visible=True)

        def _decrypt_to_progress(image: Image.Image, pw: str):
            result = _obfuscate_pil(image, pw, decrypt=True)
            return gr.update(value=result, visible=True)

        encrypt_btn.click(
            _encrypt_to_progress,
            inputs=[input_image, password],
            outputs=progress_window,
            show_progress=True,
        )
        decrypt_btn.click(
            _decrypt_to_progress,
            inputs=[input_image, password],
            outputs=progress_window,
            show_progress=True,
        )
