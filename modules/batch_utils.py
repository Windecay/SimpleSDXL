import os
import uuid
import copy
import random
import threading
import numpy as np
import gradio as gr
from PIL import Image

BATCH_EVENTS = {}
_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder_path):
    if folder_path is None:
        return []
    folder_path = str(folder_path).strip()
    if folder_path == "" or not os.path.isdir(folder_path):
        return []
    files = []
    for name in os.listdir(folder_path):
        full = os.path.join(folder_path, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in _EXTS:
            files.append(full)
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


def files_from_upload(upload_files):
    if upload_files is None:
        return []
    if isinstance(upload_files, (str, dict)):
        upload_files = [upload_files]
    if not isinstance(upload_files, list):
        return []
    files = []
    for it in upload_files:
        p = None
        if isinstance(it, str):
            p = it
        elif isinstance(it, dict):
            p = it.get("name") or it.get("path")
        else:
            p = getattr(it, "name", None)
        if p and isinstance(p, str) and os.path.exists(p):
            files.append(p)
    files = [p for p in files if os.path.splitext(p)[1].lower() in _EXTS]
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


def get_files(folder_path, upload_files):
    upload_list = files_from_upload(upload_files)
    if len(upload_list) > 0:
        return upload_list
    return list_images(folder_path)


def load_rgba(path):
    im = Image.open(path)
    im = im.convert("RGBA")
    return np.array(im)


def create_batch(batch_id=None):
    if batch_id is None:
        batch_id = str(uuid.uuid4())
    evt = BATCH_EVENTS.get(batch_id)
    if evt is None:
        evt = threading.Event()
        BATCH_EVENTS[batch_id] = evt
    return batch_id, evt


def clear_batch(batch_id):
    if not batch_id:
        return
    if batch_id in BATCH_EVENTS:
        try:
            del BATCH_EVENTS[batch_id]
        except Exception:
            pass


def stop_batch(batch_id, worker=None):
    evt = BATCH_EVENTS.get(batch_id)
    if evt is not None:
        evt.set()
    if worker is not None:
        try:
            if getattr(worker, "worker_processing", None) is not None:
                worker.worker.interrupt_processing()
        except Exception:
            pass
    return "Stopping..." if batch_id else ""


def fill_backend_meta(args_norm, state):
    try:
        if not isinstance(args_norm, list) or len(args_norm) <= 67:
            return args_norm
        backend = args_norm[67]
        if not isinstance(backend, list):
            return args_norm
        user = state.get("user") if isinstance(state, dict) else None
        preset = state.get("__preset") if isinstance(state, dict) else None
        engine_type = state.get("engine_type") if isinstance(state, dict) else None
        if engine_type is None and isinstance(state, dict):
            engine_type = state.get("default_engine", {}).get("engine_type")
        nickname = user.get_nickname() if user is not None else ""
        user_did = user.get_did() if user is not None else ""
        backend[1] = preset if preset is not None else backend[1]
        backend[3] = nickname if nickname is not None else ""
        backend[4] = user_did if user_did is not None else ""
        backend[-1] = engine_type if engine_type is not None else backend[-1]
    except Exception:
        return args_norm
    return args_norm


def refresh_scene_batch_target(state_params, current_value):
    disvisible = []
    if isinstance(state_params, dict):
        scenes = state_params.get("scene_frontend", {})
        if isinstance(scenes, dict):
            disvisible = scenes.get("disvisible", [])
    if not isinstance(disvisible, list):
        disvisible = []
    disvisible = set(disvisible)

    choices = []
    if "scene_canvas_image" not in disvisible:
        choices.append("Upload and canvas(1)")
    if "scene_input_image1" not in disvisible:
        choices.append("Upload prompt image(2)")
    if "scene_input_image2" not in disvisible:
        choices.append("Upload prompt image(3)")
    if len(choices) == 0:
        choices = ["Prompt image(2)"]

    canvas_choice = "Upload and canvas(1)"
    if canvas_choice in choices:
        value = canvas_choice
    else:
        preferred = "Prompt image(2)" if "Prompt image(2)" in choices else choices[0]
        value = current_value if current_value in choices else preferred
    return gr.update(choices=choices, value=value)


def refresh_scene_batch_accordion(state_params):
    scenes = state_params.get("scene_frontend") if isinstance(state_params, dict) else None
    if not isinstance(scenes, dict):
        return gr.update(visible=False, open=False)
    disvisible = scenes.get("disvisible", [])
    if not isinstance(disvisible, list):
        disvisible = []
    disvisible = set(disvisible)
    if "scene_batch" in disvisible:
        return gr.update(visible=False, open=False)
    visible = ("scene_canvas_image" not in disvisible) or ("scene_input_image1" not in disvisible) or ("scene_input_image2" not in disvisible)
    return gr.update(visible=visible, open=False)


def _ensure_backend_ctrl(ctrls_values, state):
    if not isinstance(ctrls_values, list):
        return ctrls_values
    backend_index = None
    for i in range(len(ctrls_values) - 1, -1, -1):
        if isinstance(ctrls_values[i], dict):
            backend_index = i
            break
    if backend_index is None:
        return ctrls_values

    backend = ctrls_values[backend_index]
    if backend is None:
        backend = {}
    if not isinstance(backend, dict):
        backend = {}
    backend = copy.deepcopy(backend)

    user = state.get("user") if isinstance(state, dict) else None
    if user is not None:
        try:
            backend["nickname"] = user.get_nickname()
        except Exception:
            backend.setdefault("nickname", "")
        try:
            backend["user_did"] = user.get_did()
        except Exception:
            backend.setdefault("user_did", "")

    if isinstance(state, dict) and "__preset" in state:
        backend["preset"] = state.get("__preset")

    engine_type = state.get("engine_type") if isinstance(state, dict) else None
    if engine_type is None and isinstance(state, dict):
        engine_type = state.get("default_engine", {}).get("engine_type")
    if engine_type is not None:
        backend["engine_type"] = engine_type

    ctrls_values = list(ctrls_values)
    ctrls_values[backend_index] = backend
    return ctrls_values


def batch_run_uov(folder_path, upload_files, seed_random, *args, get_task_with_resolution_multiplier, generate_clicked, worker, constants, html, get_welcome_image):
    if len(args) < 4:
        return
    state = args[-1]
    is_mobile = state.get("__is_mobile", False) if isinstance(state, dict) else False
    resolution_quantize_step = args[-2]
    resolution_multiplier = args[-3]
    ctrls_values = list(args[:-3])
    ctrls_values = _ensure_backend_ctrl(ctrls_values, state)

    files = get_files(folder_path, upload_files)
    if len(files) == 0:
        yield gr.update(visible=True, value=html.make_progress_html(1, "Batch: folder is empty or invalid.")), \
            gr.update(visible=True, value=get_welcome_image(is_mobile=is_mobile, is_change=True)), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
            gr.update(interactive=True), False, "Batch: folder is empty or invalid.", ""
        return

    batch_id, evt = create_batch()
    yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: 0/{len(files)}")), \
        gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
        gr.update(interactive=False), True, f"Batch started: {len(files)} files", batch_id

    base_task = get_task_with_resolution_multiplier(*ctrls_values, resolution_multiplier, resolution_quantize_step)
    base_args = copy.deepcopy(base_task.args)
    base_args = fill_backend_meta(base_args, state)

    try:
        if seed_random:
            base_args[8] = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            base_args[8] = int(base_args[8])
    except Exception:
        pass

    stopped = False
    completed = 0

    for i, path in enumerate(files):
        if evt.is_set():
            stopped = True
            break
        try:
            img = load_rgba(path)
        except Exception as e:
            yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: failed to load {os.path.basename(path)} ({e})")), \
                gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), gr.update(), gr.update(), \
                gr.update(interactive=False), True, f"Batch: failed to load {os.path.basename(path)}", batch_id
            continue

        args_i = copy.deepcopy(base_args)
        args_i[19] = img
        task = worker.AsyncTask(args=args_i)

        status = f"Batch UOV: {i + 1}/{len(files)} - {os.path.basename(path)}"
        for out in generate_clicked(task, state):
            yield (*out, gr.update(interactive=False), True, status, batch_id)
        completed = i + 1
        if evt.is_set():
            stopped = True
            break

    clear_batch(batch_id)
    if stopped:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, f"Batch stopped: {completed}/{len(files)} files", batch_id
    else:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, "Batch finished.", batch_id


def batch_run_enhance(folder_path, upload_files, seed_random, *args, get_task_with_resolution_multiplier, generate_clicked, worker, constants, html, get_welcome_image):
    if len(args) < 4:
        return
    state = args[-1]
    is_mobile = state.get("__is_mobile", False) if isinstance(state, dict) else False
    resolution_quantize_step = args[-2]
    resolution_multiplier = args[-3]
    ctrls_values = list(args[:-3])
    ctrls_values = _ensure_backend_ctrl(ctrls_values, state)

    files = get_files(folder_path, upload_files)
    if len(files) == 0:
        yield gr.update(visible=True, value=html.make_progress_html(1, "Batch: folder is empty or invalid.")), \
            gr.update(visible=True, value=get_welcome_image(is_mobile=is_mobile, is_change=True)), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
            gr.update(interactive=True), False, "Batch: folder is empty or invalid.", ""
        return

    batch_id, evt = create_batch()
    yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: 0/{len(files)}")), \
        gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
        gr.update(interactive=False), True, f"Batch started: {len(files)} files", batch_id

    base_task = get_task_with_resolution_multiplier(*ctrls_values, resolution_multiplier, resolution_quantize_step)
    base_args = copy.deepcopy(base_task.args)
    base_args = fill_backend_meta(base_args, state)

    try:
        if seed_random:
            base_args[8] = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            base_args[8] = int(base_args[8])
    except Exception:
        pass

    stopped = False
    completed = 0

    for i, path in enumerate(files):
        if evt.is_set():
            stopped = True
            break
        try:
            img = load_rgba(path)
        except Exception as e:
            yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: failed to load {os.path.basename(path)} ({e})")), \
                gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), gr.update(), gr.update(), \
                gr.update(interactive=False), True, f"Batch: failed to load {os.path.basename(path)}", batch_id
            continue

        args_i = copy.deepcopy(base_args)
        args_i[75] = img
        task = worker.AsyncTask(args=args_i)

        status = f"Batch Enhance: {i + 1}/{len(files)} - {os.path.basename(path)}"
        for out in generate_clicked(task, state):
            yield (*out, gr.update(interactive=False), True, status, batch_id)
        completed = i + 1
        if evt.is_set():
            stopped = True
            break

    clear_batch(batch_id)
    if stopped:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, f"Batch stopped: {completed}/{len(files)} files", batch_id
    else:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, "Batch finished.", batch_id


def batch_run_scene(folder_path, upload_files, target, seed_random, image_seed, backend_params, scene_theme, scene_canvas_image, scene_input_image1, scene_input_image2, scene_additional_prompt, scene_additional_prompt_2,
                    scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6,
                    scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps,
                    scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio,
                    scene_image_number, scene_video, scene_audio, scene_original_video_path, active_video_source,
                    sam3_input_video, sam3_original_video_path, sam3_mask_video, *args, get_task_with_resolution_multiplier, generate_clicked, worker, constants, html, get_welcome_image, api_params, topbar):
    if len(args) < 4:
        return
    state = args[-1]
    is_mobile = state.get("__is_mobile", False) if isinstance(state, dict) else False
    resolution_quantize_step = args[-2]
    resolution_multiplier = args[-3]
    ctrls_values = list(args[:-3])
    ctrls_values = _ensure_backend_ctrl(ctrls_values, state)

    files = get_files(folder_path, upload_files)
    if len(files) == 0:
        yield gr.update(visible=True, value=html.make_progress_html(1, "Batch: folder is empty or invalid.")), \
            gr.update(visible=True, value=get_welcome_image(is_mobile=is_mobile, is_change=True)), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
            gr.update(interactive=True), False, "Batch: folder is empty or invalid.", ""
        return

    batch_id, evt = create_batch()
    yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: 0/{len(files)}")), \
        gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(visible=False), gr.update(visible=False, size="sm"), gr.update(), gr.update(), \
        gr.update(interactive=False), True, f"Batch started: {len(files)} files", batch_id

    base_task = get_task_with_resolution_multiplier(*ctrls_values, resolution_multiplier, resolution_quantize_step)
    base_args = copy.deepcopy(base_task.args)

    fixed_seed = None
    try:
        if seed_random:
            fixed_seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            fixed_seed = int(image_seed)
    except Exception:
        fixed_seed = None
    if fixed_seed is not None:
        try:
            base_args[8] = fixed_seed
        except Exception:
            pass

    stopped = False
    completed = 0

    def _build_canvas_value(img_rgba):
        h, w = img_rgba.shape[0], img_rgba.shape[1]
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        return {"image": img_rgba, "mask": mask}

    for i, path in enumerate(files):
        if evt.is_set():
            stopped = True
            break
        try:
            img = load_rgba(path)
        except Exception as e:
            yield gr.update(visible=True, value=html.make_progress_html(1, f"Batch: failed to load {os.path.basename(path)} ({e})")), \
                gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), gr.update(), gr.update(), \
                gr.update(interactive=False), True, f"Batch: failed to load {os.path.basename(path)}", batch_id
            continue

        scene_canvas_image_v = copy.deepcopy(scene_canvas_image) if isinstance(scene_canvas_image, dict) else scene_canvas_image
        scene_input_image1_v = scene_input_image1
        scene_input_image2_v = scene_input_image2
        if target == "Upload and canvas(1)":
            scene_canvas_image_v = _build_canvas_value(img)
        elif target == "Upload prompt image(3)":
            scene_input_image2_v = img
        else:
            scene_input_image1_v = img

        bp = {} if backend_params is None else copy.deepcopy(backend_params)
        try:
            topbar.process_before_generation(
                state, False if fixed_seed is not None else seed_random, fixed_seed if fixed_seed is not None else image_seed,
                bp, scene_theme, scene_canvas_image_v, scene_input_image1_v, scene_input_image2_v,
                scene_additional_prompt, scene_additional_prompt_2,
                scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5,
                scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10,
                scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4,
                scene_aspect_ratio, scene_image_number,
                scene_video, scene_audio, scene_original_video_path, active_video_source,
                sam3_input_video, sam3_original_video_path, sam3_mask_video
            )
        except Exception:
            bp = {} if backend_params is None else copy.deepcopy(backend_params)
        backend_norm = api_params.normalization_backend(bp)

        args_i = copy.deepcopy(base_args)
        args_i[67] = backend_norm
        args_i = fill_backend_meta(args_i, state)
        if fixed_seed is not None:
            try:
                args_i[8] = fixed_seed
            except Exception:
                pass

        task = worker.AsyncTask(args=args_i)
        status = f"Batch Scene: {i + 1}/{len(files)} - {os.path.basename(path)}"
        for out in generate_clicked(task, state):
            yield (*out, gr.update(interactive=False), True, status, batch_id)
        completed = i + 1
        if evt.is_set():
            stopped = True
            break

    clear_batch(batch_id)
    if stopped:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, f"Batch stopped: {completed}/{len(files)} files", batch_id
    else:
        yield gr.update(visible=False), \
            gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), \
            gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), \
            gr.update(visible=True, interactive=True), False, "Batch finished.", batch_id
