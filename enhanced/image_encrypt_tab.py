import functools
import math
import os
from typing import Tuple

import gradio as gr
import numpy as np
from PIL import Image

import modules.config as config
import modules.gradio_hijack as grh
import modules.util as util


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


def add_image_encrypt_tab(
    progress_window,
    progress_gallery,
    gallery,
    progress_video,
    comparison_box,
    compare_btn,
    comparison_state,
    state_topbar,
    state_is_generating,
    image_toolbox,
    gallery_index,
    output_format,
):
    with gr.Tab(label="Image Encrypt", id="image_encrypt_tab", visible=True):
        with gr.Column():
            input_image = grh.Image(label="Input Image", source="upload", type="pil")
            password = gr.Textbox(label="Password", value="", placeholder="0100")
            save_decrypted = gr.Checkbox(label="Save decrypted image", value=True, container=False)
            with gr.Row():
                encrypt_btn = gr.Button(value="Encrypt")
                decrypt_btn = gr.Button(value="Decrypt")
            gr.HTML('<div style="font-size: 12px; opacity: 0.75;">Source: https://dfqtphx.netlify.app/</div>')

        def _encrypt_to_progress(image: Image.Image, pw: str, state_params: dict, is_generating: bool, current_comparison_state: bool):
            if is_generating:
                return (
                    current_comparison_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state_params,
                )

            result = _obfuscate_pil(image, pw, decrypt=False)
            state_params = dict(state_params or {})
            state_params["gallery_state"] = "preview"
            return (
                False,
                gr.update(visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=([result] if result is not None else []), visible=True),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(visible=False, size="sm"),
                gr.update(visible=False),
                gr.update(),
                state_params,
            )

        def _decrypt_to_progress(
            image: Image.Image,
            pw: str,
            should_save: bool,
            output_format: str,
            state_params: dict,
            is_generating: bool,
            current_comparison_state: bool,
            current_gallery_choice: str | None,
        ):
            if is_generating:
                return (
                    current_comparison_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state_params,
                )

            result = _obfuscate_pil(image, pw, decrypt=True)
            state_params = dict(state_params or {})
            gallery_index_update = gr.update()
            if should_save and result is not None:
                try:
                    import modules.flags as flags

                    user = state_params.get("user")
                    user_did = user.get_did() if user is not None else None
                    output_root = config.get_user_path_outputs(user_did)

                    target_format = str(output_format or "").strip().lower()
                    if target_format not in flags.OutputFormat.list():
                        target_format = config.default_output_format

                    _, abs_path, _ = util.generate_temp_filename(folder=output_root, extension=target_format)
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    if target_format == flags.OutputFormat.PNG.value:
                        result.save(abs_path, format="PNG")
                    elif target_format == flags.OutputFormat.JPEG.value:
                        if result.mode == "RGBA":
                            background = Image.new("RGB", result.size, (255, 255, 255))
                            background.paste(result, mask=result.getchannel("A"))
                            background.save(abs_path, format="JPEG", quality=95, optimize=True, progressive=True)
                        else:
                            result.convert("RGB").save(abs_path, format="JPEG", quality=95, optimize=True, progressive=True)
                    elif target_format == flags.OutputFormat.WEBP.value:
                        result.save(abs_path, format="WEBP", quality=95, lossless=False)
                    else:
                        result.save(abs_path, format=target_format)

                    try:
                        import enhanced.gallery as gallery_util

                        max_per_page = state_params.get("__max_per_page", 18)
                        max_catalog = state_params.get("__max_catalog", config.default_image_catalog_max_number)
                        engine_type = state_params.get("engine_type", "image")

                        saved_dirname = os.path.basename(os.path.dirname(abs_path))
                        if len(saved_dirname) >= 2 and saved_dirname[:2] == "20":
                            output_choice = saved_dirname[2:]
                            gallery_util.refresh_images_catalog(output_choice, True, user_did=user_did)

                        output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog, user_did, engine_type)
                        state_params.update({"__output_list": output_list})
                        state_params.update({"__finished_nums_pages": f"{finished_nums},{finished_pages}"})
                        gallery_index_update = gr.update(choices=output_list)
                    except Exception:
                        pass
                except Exception:
                    pass
            state_params["gallery_state"] = "preview"
            return (
                False,
                gr.update(visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=([result] if result is not None else []), visible=True),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(visible=False, size="sm"),
                gr.update(visible=False),
                gallery_index_update,
                state_params,
            )

        encrypt_btn.click(
            _encrypt_to_progress,
            inputs=[input_image, password, state_topbar, state_is_generating, comparison_state],
            outputs=[comparison_state, comparison_box, progress_window, progress_gallery, progress_video, gallery, compare_btn, image_toolbox, gallery_index, state_topbar],
            show_progress=True,
            queue=False,
        )
        decrypt_btn.click(
            _decrypt_to_progress,
            inputs=[input_image, password, save_decrypted, output_format, state_topbar, state_is_generating, comparison_state, gallery_index],
            outputs=[comparison_state, comparison_box, progress_window, progress_gallery, progress_video, gallery, compare_btn, image_toolbox, gallery_index, state_topbar],
            show_progress=True,
            queue=False,
        )
