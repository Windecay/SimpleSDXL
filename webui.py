import io
import base64
import numpy as np
import gradio as gr
import os
import json
import time
import re
import random
import tempfile
import wave

try:
    import gradio.processing_utils as _gr_processing_utils

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
except Exception:
    pass
import shared
import modules.config
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import copy
import args_manager
import ldm_patched.modules.model_management as model_management

from extras.inpaint_mask import SAMOptions
from PIL import Image
from modules.sdxl_styles import legal_style_names, fooocus_expansion
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import resize_image
import modules.util as util
from modules.meta_parser import switch_scene_theme, switch_scene_theme_select, switch_scene_theme_ready_to_gen, get_welcome_image, describe_prompt_for_scene, get_auto_candidate, extract_scene_image

import comfy.comfy_version as comfy_version
import enhanced.gallery as gallery_util
import enhanced.topbar  as topbar
import enhanced.toolbox  as toolbox
import enhanced.translator  as translator
import enhanced.version as version
import enhanced.wildcards as wildcards
import enhanced.simpleai as simpleai
import enhanced.comfy_task as comfy_task
import enhanced.all_parameters as ads
import simpleai_base.api_params as api_params
from enhanced.simpleai import comfyd, p2p_task 
from enhanced.minicpm import MiniCPM, minicpm
from enhanced.inference_artist import get_artist_tags_string
import modules.model_loader as model_loader
import enhanced.qwen_multiangle as qwen_multiangle
import enhanced.qwen_anglelight as qwen_anglelight
import enhanced.transfer_style_gallery as transfer_style_gallery
import enhanced.sam3_video_mask as sam3_video_mask
import logging
logger = logging.getLogger(__name__)

START_TIMESTAMP = time.time()

def get_cookie_value(cookie_string, key):
    pattern = rf'{key}=([^;]+)'
    match = re.search(pattern, cookie_string)
    if match:
        return match.group(1)
    return None

def get_start_timestamp(request: gr.Request):
    global START_TIMESTAMP

    online_users, domain_online_nodes, domain_online_users, new_msg_number = 0, 0, 0, 0
    if "cookie" in request.headers:
        sid = get_cookie_value(request.headers["cookie"], "aitoken")
        if sid:
            online_users, domain_online_nodes, domain_online_users, new_msg_number = shared.token.log_access(sid)
            #node_all, usesr_all, new_msg = shared.token.get_global_status(sid,0)
        
    qsize = worker.get_task_size()
    vram_ram_info = model_management.get_vram_ram_used()
    if new_msg_number>0:
        print(f'new messages: {shared.token.get_global_msg_all()}')
    return f'{START_TIMESTAMP},{qsize},{vram_ram_info[0]},{vram_ram_info[1]},{vram_ram_info[2]},{vram_ram_info[3]},{online_users},{domain_online_users},{domain_online_nodes}'

def get_wildcards_list(request: gr.Request):
    wildcard_list = wildcards.get_wildcards_samples(trans=False)
    wildcard_list = [w[0] for w in wildcard_list]
    wildcard_list = ','.join(wildcard_list)
    return wildcard_list

def get_task(*args):
    args = list(args)
    args.pop(0)
    args = api_params.normalization(args, modules.config.default_max_lora_number, modules.config.default_controlnet_image_count, modules.config.default_enhance_tabs)
    return worker.AsyncTask(args=args)

def get_task_with_resolution_multiplier(*args):
    args = list(args)
    args.pop(0)
    resolution_quantize_step = args.pop() if len(args) > 0 else 8
    resolution_multiplier = args.pop() if len(args) > 0 else 1.0
    args = api_params.normalization(args, modules.config.default_max_lora_number, modules.config.default_controlnet_image_count, modules.config.default_enhance_tabs)

    try:
        m = float(resolution_multiplier)
    except Exception:
        m = 1.0

    if m > 1.0:
        try:
            m = max(1.0, min(2.0, m))
            try:
                step = int(resolution_quantize_step)
            except Exception:
                step = 8
            if step not in [8, 16, 32, 64]:
                step = 8

            aspect_ratios_index = api_params.all_args.index('aspect_ratios_selection')
            overwrite_width_index = api_params.all_args.index('overwrite_width')
            overwrite_height_index = api_params.all_args.index('overwrite_height')

            overwrite_width = int(args[overwrite_width_index]) if args[overwrite_width_index] is not None else -1
            overwrite_height = int(args[overwrite_height_index]) if args[overwrite_height_index] is not None else -1

            base_w = overwrite_width
            base_h = overwrite_height
            if base_w <= 0 or base_h <= 0:
                try:
                    import re
                    raw = str(args[aspect_ratios_index] or "")
                    raw = raw.split(',', 1)[0]
                    m2 = re.search(r'(\d+)\D+(\d+)', raw.replace('×', 'x'))
                    if m2:
                        base_w = int(m2.group(1))
                        base_h = int(m2.group(2))
                except Exception:
                    base_w = -1
                    base_h = -1

            if base_w > 0 and base_h > 0:
                def _quantize(v):
                    v = int(round(float(v) / float(step)) * step)
                    if v <= 0:
                        v = step
                    return v

                args[overwrite_width_index] = _quantize(base_w * m)
                args[overwrite_height_index] = _quantize(base_h * m)
        except Exception:
            pass

    return worker.AsyncTask(args=args)

def refresh_files_clicked(state_params, use_model_filter: bool = True, show_info: bool = True):
    engine = state_params.get('engine', 'Fooocus') if isinstance(state_params, dict) else 'Fooocus'
    task_method = state_params.get('task_method', None) if isinstance(state_params, dict) else None
    model_filenames, lora_filenames, vae_filenames = modules.config.update_files(engine, task_method, use_model_filter=use_model_filter)
    if show_info:
        try:
            gr.Info(f"[RefreshFiles] Models={len(model_filenames)} LoRAs={len(lora_filenames)}")
        except Exception as e:
            logger.info(f"[RefreshFiles] gr.Info failed: {e}")
    results = [gr.update(choices=model_filenames)]
    results += [gr.update(choices=['None'] + model_filenames)]
    results += [gr.update(choices=[flags.default_vae] + vae_filenames)]
    for _ in range(4):
        results.append(gr.update(choices=['None'] + lora_filenames))
        results.append(gr.update(interactive=True))
    for _ in range(modules.config.default_max_lora_number):
        results += [gr.update(interactive=True), gr.update(choices=['None'] + lora_filenames), gr.update()]
    return results

def generate_clicked(task: worker.AsyncTask, state):
    user_did = None
    try:
        if isinstance(state, dict) and "user" in state and state["user"] is not None:
            user_did = state["user"].get_did()
    except Exception:
        user_did = None

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    if len(task.args) == 0:
        return
    is_mobile = state["__is_mobile"]
    is_fooocus = state["engine"] == 'Fooocus'
    task_meta = f"task_id={getattr(task, 'task_id', None)}, user_did={user_did}, task_class={getattr(task, 'task_class', None)}, task_name={getattr(task, 'task_name', None)}, task_method={getattr(task, 'task_method', None)}"

    # outputs=[progress_html, progress_window, progress_gallery, progress_video, gallery]
    # if "absent_model" in state and state["absent_model"]:
    #     yield gr.update(visible=False), \
    #         gr.update(visible=True, value=get_welcome_image(is_mobile=is_mobile, is_change=True)), \
    #         gr.update(visible=False, value=None), \
    #         gr.update(visible=False), \
    #         gr.update(visible=False)
        # return

    MAX_WAIT_TIME = 1800
    POLL_INTERVAL = 0.1

    worker.add_task(task)
    qsize = worker.get_task_size()
    MAX_LOOP_NUM = qsize
    last_update_time = time.time()
    loop_num = 0
    ready_flag = False
    queue_start_time = time.time()
    logged_queue_wait = False
    logger.info(f"[Generate] enqueue: qsize={qsize}, {task_meta}")
    try:
        while qsize > 0:
            current_time = time.time()
            if len(task.yields) > 0 or len(task.results) > 0 or task.processing:
                ready_flag = True
                logger.info(f"[Generate] queue_exit(activity): waited={current_time - queue_start_time:.2f}s, qsize={qsize}, {task_meta}")
                break
            if (current_time - MAX_WAIT_TIME*loop_num - last_update_time) < MAX_WAIT_TIME:
                if (not logged_queue_wait) and (current_time - queue_start_time) >= 5.0:
                    logged_queue_wait = True
                    logger.warning(f"[Generate] queue_wait: waited={current_time - queue_start_time:.2f}s, qsize={qsize}, processing_id={worker.get_processing_id()}, {task_meta}")
                yield gr.update(visible=True, value=modules.html.make_progress_html(1, f'生图任务已入队列({qsize})，请等待...')), \
                    gr.update(visible=True, value=get_welcome_image(is_mobile=is_mobile, is_change=True)), \
                    gr.update(visible=False, value=None), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    False, \
                    gr.update(visible=False), \
                    gr.update(visible=False, size='sm'), \
                    gr.update(interactive=False), \
                    gr.update(interactive=False)
                if qsize<=1 or worker.get_processing_id() == task.task_id:
                    ready_flag = True
                    logger.info(f"[Generate] queue_exit(turn): waited={current_time - queue_start_time:.2f}s, qsize={qsize}, processing_id={worker.get_processing_id()}, {task_meta}")
                    break
            else:
                loop_num += 1
                if loop_num > MAX_LOOP_NUM:
                    logger.info(f"[Generate] queue_restart_worker: loop_num={loop_num}, max_loop={MAX_LOOP_NUM}, {task_meta}")
                    worker.restart(task)
                    break
            time.sleep(POLL_INTERVAL)
            qsize = worker.get_task_size()
    except GeneratorExit:
        logger.warning(f"[Generate] client_disconnected(queue): waited={time.time() - queue_start_time:.2f}s, qsize={qsize}, {task_meta}")
        raise
    except BaseException:
        logger.exception(f"[Generate] error(queue): waited={time.time() - queue_start_time:.2f}s, qsize={qsize}, {task_meta}")
        raise
    
    execution_start_time = time.perf_counter()
    finished = False
    ready_flag = True if qsize<=1 else ready_flag
    MAX_WAIT_TIME = 1800 if task.content_type == 'image' else 7200
    POLL_INTERVAL = 0.08
    in_progress = False
    local_start_time = time.time()
    last_heartbeat_time = local_start_time
    HEARTBEAT_INTERVAL = 1.0
    UNLOCK_CONTROLS_AFTER = 5.0
    logged_controls_unlock = False
    logged_backend_ready_wait = False
    logged_first_yield = False
    yields_processed = 0
    logger.info(f"[Generate] start: qsize={qsize}, ready_flag={ready_flag}, {task_meta}")

    last_update_time = time.time()

    preview_cache = []
    preview_cache_index = 0
    last_preview_title = ""
    last_preview_percentage = 0
    last_preview_frame_time = local_start_time
    next_preview_ui_time = local_start_time
    last_preview_shown_title = ""
    last_preview_shown_percentage = 0
    waiting_for_new_step_frame = False
    backend_ready = False
    preview_interval = 1.0 / 8.0

    try:
        while not finished:
            current_time = time.time()
            if (current_time - last_update_time > MAX_WAIT_TIME) or not ready_flag:
                yield gr.update(visible=True, value=modules.html.make_progress_html(0, '生图任务已超时!')), \
                    gr.update(visible=True), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    False, \
                    gr.update(visible=False), \
                    gr.update(visible=False, size='sm'), \
                    gr.update(interactive=True), \
                    gr.update(interactive=True)
                logger.error(f"[Generate] timeout: max_wait={MAX_WAIT_TIME}, ready_flag={ready_flag}, last_update_time={last_update_time}, {task_meta}")
                task.last_stop = 'stop'
                worker.worker.stop_processing(task, 0, 'timeout')
                if (task.processing):
                    logger.error(f"[Generate] timeout_interrupt: {task_meta}")
                    worker.worker.interrupt_processing()
                yield gr.update(visible=False), \
                    gr.update(visible=True), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    False, \
                    gr.update(visible=False), \
                    gr.update(visible=False, size='sm'), \
                    gr.update(interactive=True), \
                    gr.update(interactive=True)
                break

            time.sleep(POLL_INTERVAL)

            controls_unlocked = backend_ready or ((current_time - local_start_time) >= UNLOCK_CONTROLS_AFTER)
            if controls_unlocked and (not backend_ready) and (not logged_controls_unlock):
                logged_controls_unlock = True
                logger.warning(f"[Generate] controls_unlocked_by_timeout: unlock_after={UNLOCK_CONTROLS_AFTER}s, {task_meta}")

            if (not backend_ready) and (not logged_backend_ready_wait) and ((current_time - local_start_time) >= 15.0):
                logged_backend_ready_wait = True
                logger.warning(f"[Generate] backend_ready_delayed: waited={current_time - local_start_time:.2f}s, yields_len={len(task.yields)}, processing={getattr(task, 'processing', None)}, {task_meta}")

            if len(preview_cache) > 1 and current_time >= next_preview_ui_time:
                head_flag = task.yields[0][0] if len(task.yields) > 0 else None
                head_preview_image_none = False
                if head_flag == 'preview':
                    try:
                        head_preview_image_none = task.yields[0][1][2] is None
                    except Exception:
                        head_preview_image_none = False

                can_rotate_now = (len(task.yields) == 0) or (head_flag != 'preview') or head_preview_image_none
                if can_rotate_now:
                    preview_cache_index = (preview_cache_index + 1) % len(preview_cache)
                    cached_image = preview_cache[preview_cache_index]
                    last_preview_frame_time = current_time
                    next_preview_ui_time += preview_interval
                    if next_preview_ui_time <= current_time:
                        next_preview_ui_time = current_time + preview_interval
                    yield gr.update(visible=True, value=modules.html.make_progress_html(last_preview_percentage, last_preview_title)), \
                        gr.update(visible=True, value=cached_image), \
                        gr.update(), \
                        gr.update(visible=False), \
                        gr.update(visible=False), \
                        False, \
                        gr.update(visible=False), \
                        gr.update(visible=False, size='sm'), \
                        gr.update(interactive=True), \
                        gr.update(interactive=True)
                    continue

            if len(task.yields) > 0:
                flag, product = task.yields.pop(0)
                yields_processed += 1
                in_progress = True
                if not logged_first_yield:
                    logged_first_yield = True
                    logger.info(f"[Generate] first_yield: delay={current_time - local_start_time:.2f}s, flag={flag}, {task_meta}")

                if flag == 'status':
                    if product == 'backend_ready':
                        backend_ready = True
                        logger.info(f"[Generate] backend_ready: delay={current_time - local_start_time:.2f}s, {task_meta}")
                        yield gr.update(visible=True, value=modules.html.make_progress_html(1, '任务准备开始，加载模型...')), \
                            gr.update(), \
                            gr.update(), \
                            gr.update(visible=False), \
                            gr.update(visible=False), \
                            False, \
                            gr.update(visible=False), \
                            gr.update(visible=False, size='sm'), \
                            gr.update(interactive=True), \
                            gr.update(interactive=True)

                if flag == 'preview':
                    last_update_time = current_time
                    percentage, title, image = product

                    title_changed = title != last_preview_title
                    if title != last_preview_title:
                        last_preview_title = title
                        waiting_for_new_step_frame = True
                        if task.content_type != 'video':
                            preview_cache = []
                            preview_cache_index = 0

                    if image is not None:
                        if waiting_for_new_step_frame:
                            preview_cache = []
                            preview_cache_index = 0
                            waiting_for_new_step_frame = False

                        preview_cache.append(image)

                    last_preview_percentage = percentage
                    image_to_show = image
                    if image_to_show is None and len(preview_cache) > 0 and not waiting_for_new_step_frame:
                        preview_cache_index = (preview_cache_index + 1) % len(preview_cache)
                        image_to_show = preview_cache[preview_cache_index]
                    if image_to_show is not None:
                        last_preview_frame_time = current_time

                    should_yield_preview = False
                    if title_changed:
                        should_yield_preview = True
                    elif image_to_show is not None:
                        should_yield_preview = current_time >= next_preview_ui_time
                    else:
                        should_yield_preview = (
                            (percentage != last_preview_shown_percentage or title != last_preview_shown_title)
                            and current_time >= next_preview_ui_time
                        )

                    if not should_yield_preview:
                        continue

                    if title_changed:
                        next_preview_ui_time = current_time + preview_interval
                    else:
                        next_preview_ui_time += preview_interval
                        if next_preview_ui_time <= current_time:
                            next_preview_ui_time = current_time + preview_interval
                    last_preview_shown_percentage = percentage
                    last_preview_shown_title = title
                    yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                        gr.update(visible=True, value=image_to_show) if image_to_show is not None else gr.update(), \
                        gr.update(), \
                        gr.update(visible=False), \
                        gr.update(visible=False), \
                        False, \
                        gr.update(visible=False), \
                        gr.update(visible=False, size='sm'), \
                        gr.update(interactive=controls_unlocked), \
                        gr.update(interactive=controls_unlocked)
                if flag == 'results':
                    preview_cache = []
                    last_update_time = current_time

                    yield gr.update(visible=True), \
                        gr.update(visible=True), \
                        gr.update(visible=True, value=product), \
                        gr.update(visible=False), \
                        gr.update(visible=False), \
                        False, \
                        gr.update(visible=False), \
                        gr.update(visible=False, size='sm'), \
                        gr.update(interactive=True), \
                        gr.update(interactive=True)
                if flag == 'finish':
                    preview_cache = []
                    if not args_manager.args.disable_enhance_output_sorting and is_fooocus:
                        product = sort_enhance_images(product, task)

                    has_video = False
                    video_path = None
                    for path in product:
                        if isinstance(path, str) and path.lower().endswith(('.mp4', '.webm')):
                            has_video = True
                            video_path = path
                            break

                    yield gr.update(visible=False), \
                        gr.update(visible=False, value=get_welcome_image(is_mobile=is_mobile)), \
                        gr.update(visible=False if has_video else True, value=product), \
                        gr.update(visible=True if has_video else False, value=video_path), \
                        gr.update(visible=False), \
                        False, \
                        gr.update(visible=False), \
                        gr.update(visible=False, size='sm'), \
                        gr.update(interactive=True), \
                        gr.update(interactive=True)
                    finished = True

                    # delete Fooocus temp images, only keep gradio temp images
                    if args_manager.args.disable_image_log:
                        for filepath in product:
                            if isinstance(filepath, str) and os.path.exists(filepath):
                                os.remove(filepath)

            elif len(preview_cache) > 1 and current_time >= next_preview_ui_time:
                preview_cache_index = (preview_cache_index + 1) % len(preview_cache)
                cached_image = preview_cache[preview_cache_index]
                last_preview_frame_time = current_time
                next_preview_ui_time += preview_interval
                if next_preview_ui_time <= current_time:
                    next_preview_ui_time = current_time + preview_interval

                yield gr.update(visible=True, value=modules.html.make_progress_html(last_preview_percentage, last_preview_title)), \
                    gr.update(visible=True, value=cached_image), \
                    gr.update(), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    False, \
                    gr.update(visible=False), \
                    gr.update(visible=False, size='sm'), \
                    gr.update(interactive=True), \
                    gr.update(interactive=True)
            elif (current_time - last_heartbeat_time) >= HEARTBEAT_INTERVAL:
                last_heartbeat_time = current_time
                if in_progress:
                    continue
                title = '任务准备中，加载模型...' if not backend_ready else '任务进行中...'
                yield gr.update(visible=True, value=modules.html.make_progress_html(max(last_preview_percentage, 1), title)), \
                    gr.update(), \
                    gr.update(), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    False, \
                    gr.update(visible=False), \
                    gr.update(visible=False, size='sm'), \
                    gr.update(interactive=controls_unlocked), \
                    gr.update(interactive=controls_unlocked)
    except GeneratorExit:
        logger.warning(f"[Generate] client_disconnected(running): backend_ready={backend_ready}, in_progress={in_progress}, yields_processed={yields_processed}, {task_meta}")
        raise
    except BaseException:
        logger.exception(f"[Generate] error(running): backend_ready={backend_ready}, in_progress={in_progress}, yields_processed={yields_processed}, {task_meta}")
        raise
    finally:
        execution_time = time.perf_counter() - execution_start_time
        logger.info(f"[Generate] end: finished={finished}, backend_ready={backend_ready}, in_progress={in_progress}, yields_processed={yields_processed}, exec_s={execution_time:.2f}, {task_meta}")

    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version, outpaint, state):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field
    # Flux inpaint_strength: 普通重绘0.7 外扩0.85 换物重绘应该0.85附近 提升细节0.5

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.2
        ]

    if inpaint_engine_version == 'empty':
        backend_engine = state.get('backend_engine', 'Z-image')
        if backend_engine == 'Z-image':
            task_method = 'z_image_turbo_aio_cn'
        else:
            task_method = 'SDXL' if 'task_method' not in state else state["task_method"]
        inpaint_engine_version = modules.flags.default_inpaint_engine_versions(task_method)
    
    engine = 'Fooocus' if 'engine' not in state else state['engine']
    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0 if engine=='Fooocus' else 1.0, 0.2
        ]
    
    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0 if engine=='Fooocus' else 1.0 if len(outpaint)>0 else 1.0, 1.0 if len(outpaint) > 0 else 0.618
    ]

def enhance_inpaint_mode_change(mode, inpaint_engine_version, state):
    assert mode in modules.flags.inpaint_options

    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            False, 'None', 0.5, 0.2
        ]

    if inpaint_engine_version == 'empty':
        backend_engine = state.get('backend_engine', 'Z-image')
        if backend_engine == 'Z-image':
            task_method = 'z_image_turbo_aio_cn'
        else:
            task_method = 'SDXL' if 'task_method' not in state else state["task_method"]
        inpaint_engine_version = modules.flags.default_inpaint_engine_versions(task_method)

    if mode == modules.flags.inpaint_option_modify:
        return [
            True, inpaint_engine_version, 1.0, 0.2
        ]

    return [
        False, inpaint_engine_version, 1.0, 0.618
    ]
def check_generating_state(state_is_generating=None, pending_tasks=None, worker_processing=None):
    if state_is_generating is None:
        state_is_generating = False
    if pending_tasks is None:
        import modules.async_worker
        pending_tasks = modules.async_worker.pending_tasks
    if worker_processing is None:
        import modules.async_worker
        worker_processing = modules.async_worker.worker_processing is not None
    return state_is_generating or pending_tasks > 0 or worker_processing

from extras.media_normalize import stash_preview_image, stash_preview_image_only, stash_preview_sketch, compose_full_sketch, compose_full_mask

reload_javascript()

title = f'{version.branch}-让创作如此轻松! Make creation a breeze!'

shared.gradio_root = gr.Blocks(title=title).queue(concurrency_count=5)

get_local_url = f'http://{args_manager.args.listen}:{args_manager.args.port}{args_manager.args.webroot}'
logo_imag_path = os.path.abspath(f'./presets/image/simpai_logo.jpg')
logo_imag_url = f'/file={logo_imag_path}'

with shared.gradio_root:
    state_topbar = gr.State({})
    cached_input_image = gr.State(None)
    params_backend = gr.State({})
    system_params = gr.JSON({}, visible=False)
    gallery_index_stat = gr.Textbox(value='', visible=False)
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    state_is_generating = gr.State(False)
    comparison_state = gr.State(False)
    random_aspect_ratio_state = gr.State(None)
    scene_video_backup = gr.State(None)
    scene_audio_backup = gr.State(None)
    scene_original_video_path = gr.State(None)
    scene_original_video_backup = gr.State(None)
    active_video_source = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group(elem_id='main_content'):
                with gr.Row(elem_id="topbar_row"):
                    start_timestamp = gr.Textbox(visible=False)
                    bar_store_button = gr.Button(value='PresetStore', size='sm', min_width=50, elem_id='bar_store', elem_classes='bar_store')
                    bar_buttons = []
                    for i in range(shared.BUTTON_NUM):
                        bar_buttons.append(gr.Button(value='default' if i==0 else '', size='sm', visible=True, min_width=40, elem_id=f'bar{i}', elem_classes='bar_button'))
                        if i == 5:
                            gr.HTML(value="", elem_classes="topbar_line_break")
                    shared.gradio_root.load(get_start_timestamp, outputs=start_timestamp, queue=False)
                    shared.gradio_root.load(get_wildcards_list, outputs=start_timestamp, queue=False)
                with gr.Row(visible=False, elem_classes='preset_store') as preset_store:
                    preset_store_list = gr.Dataset(label="My preset store: Click on the preset in store to append it to the navigation. If it is already on, it will be automatically removed.", components=[gallery_index_stat], samples=topbar.get_preset_samples(), visible=True, samples_per_page=10000, type='index')

                missing_model_modal = gr.Box(
                    visible=False,
                    elem_id="missing_model_modal",
                    elem_classes=["modal", "missing-model-modal"])
                with missing_model_modal:
                    modal_content = gr.Column(
                        elem_classes=["modal-content"],
                        elem_id="missing_model_modal_content",
                        scale=1,
                        min_width=800
                    )
                    with modal_content:
                        with gr.Row(elem_id="missing_model_modal_header"):
                            missing_model_title = gr.Markdown("### 以下模型文件缺失，请点击下载按钮获取：", elem_id="missing_model_modal_handle")
                            missing_model_minimize_btn = gr.Button(value="▁", size="sm", min_width=40, elem_id="missing_model_modal_minimize_btn")
                            close_missing_model_btn = gr.Button(value="✕", size="sm", min_width=40, elem_id="missing_model_modal_close_btn")

                        missing_model_total_progress = gr.HTML(value="", visible=False, elem_id="missing_model_total_progress")

                        dataframe_container = gr.Box(elem_id="missing_model_dataframe_container")
                        with dataframe_container:
                            missing_model_list = gr.Dataframe(
                                headers=["模型名称", "模型大小", "操作"],
                                datatype=["str", "str", "str"],
                                value=[],
                                interactive=False,
                                elem_id="missing_model_list",
                                col_count=3,
                                row_count=10,
                                max_rows=200,
                                overflow_row_behaviour="scroll")

                        with gr.Row(elem_id="missing_model_modal_actions"):
                            missing_model_btn = gr.Button("补全所选预置包", visible=False)

                def _make_missing_model_progress_html(percent):
                    try:
                        p = float(percent)
                    except Exception:
                        p = 0.0
                    p = max(0.0, min(100.0, p))
                    p_int = int(round(p))
                    p_txt = f"{p:.1f}%"
                    return f'<div class="mm-progress"><progress value="{p_int}" max="100"></progress><span class="mm-progress-label">总下载</span><span class="mm-progress-percent">{p_txt}</span></div>'

                def check_and_show_missing_models(button_value, state_params):
                    """检查模型是否缺失并显示提示窗口"""

                    if ads.get_user_default("no_model_modal_checkbox", state_params, False):
                        return [gr.update(visible=False), gr.update(value=[]), gr.update(visible=False, value=""), gr.update(visible=False)]

                    preset_name = button_value.replace('⬇', '').strip()
                    if not preset_name:
                        return [gr.update(visible=False), gr.update(value=[]), gr.update(visible=False, value=""), gr.update(visible=False)]
                    missing_models = model_loader.get_missing_model_list(preset_name)

                    if missing_models:
                        total_size = 0
                        display_data = []
                        for cata, path_file, human_size, url, size in missing_models:
                            model_name = os.path.basename(path_file)
                            display_data.append([model_name, human_size, f"下载 {model_name}"])
                            try:
                                total_size += int(size or 0)
                            except Exception:
                                pass

                        progress_value = ""
                        progress_visible = False
                        if total_size > 0:
                            progress_value = _make_missing_model_progress_html(0)
                            progress_visible = True
                        return [gr.update(visible=True),
                                gr.update(value=display_data),
                                gr.update(visible=progress_visible, value=progress_value),
                                gr.update(visible=True)]
                    else:
                        return [gr.update(visible=False), gr.update(value=[]), gr.update(visible=False, value=""), gr.update(visible=False)]

                def download_models(state_params):
                    preset_name = state_params.get('__preset', '')
                    empty_buttons_update = [gr.update() for _ in range(len(bar_buttons))]

                    if not preset_name:
                        yield [gr.update(visible=True), gr.update(), gr.update(visible=False, value="")] + empty_buttons_update
                        return

                    user_session = state_params.get('__session', '')
                    ua_hash = state_params.get('ua_hash', '')

                    user_did = shared.token.check_sstoken_and_get_did(user_session, ua_hash) if hasattr(shared.token, 'check_sstoken_and_get_did') else None
                    is_guest = not user_did or (hasattr(shared.token, 'is_guest') and shared.token.is_guest(user_did))

                    if is_guest:
                        gr.Info("游客模式下无法下载模型，请使用外置的模型管理器补全")
                        yield [gr.update(visible=False), gr.update(), gr.update(visible=False, value="")] + empty_buttons_update
                        return

                    gr.Info(f"开始下载预置包的模型: {preset_name}，请耐心等待...可于控制台查看下载进度")
                    model_loader.download_model_files(preset_name, user_did=user_did, async_task=True)

                    while True:
                        missing_models = model_loader.get_missing_model_list(preset_name)
                        if not missing_models:
                            break

                        total_current = 0
                        total_size = 0
                        has_error = False
                        has_in_progress = False
                        display_data = []
                        for cata, path_file, human_size, url, preset_size in missing_models:
                            model_name = os.path.basename(path_file)
                            status = model_loader.get_download_status(model_name)
                            try:
                                total_size += int(preset_size or 0)
                            except Exception:
                                pass
                            if status:
                                if "error" in status:
                                    has_error = True
                                    err_msg = status.get("error", "")
                                    action_text = f"Error: {err_msg}" if err_msg else "Error"
                                    try:
                                        total_current += int(status.get("current", 0) or 0)
                                    except Exception:
                                        pass
                                else:
                                    has_in_progress = True
                                    percent = status['percent']
                                    action_text = f"Downloading: {percent:.1f}%"
                                    try:
                                        total_current += int(status.get("current", 0) or 0)
                                        total_size += int(status.get("total", 0) or 0)
                                    except Exception:
                                        pass
                            else:
                                action_text = f"下载 {model_name}"
                            display_data.append([model_name, human_size, action_text])

                        percent_total = 0.0 if total_size <= 0 else max(0.0, min(100.0, (total_current / total_size) * 100.0))
                        progress_html = _make_missing_model_progress_html(percent_total)
                        yield [gr.update(visible=True), gr.update(value=display_data), gr.update(visible=True, value=progress_html)] + empty_buttons_update
                        if has_error and (not has_in_progress):
                            return
                        time.sleep(1)

                    nav_updates = topbar.refresh_nav_bars(state_params)
                    button_updates = nav_updates[1 : 1 + len(bar_buttons)]
                    yield [gr.update(visible=False), gr.update(value=[]), gr.update(visible=False, value="")] + button_updates

                def close_missing_model_modal():
                    return gr.update(visible=False)

                close_missing_model_btn.click(close_missing_model_modal, outputs=missing_model_modal)
                missing_model_minimize_btn.click(
                    fn=None,
                    _js="""() => {
                        const app = (typeof gradioApp === 'function') ? gradioApp() : document;
                        const content = app.getElementById('missing_model_modal_content');
                        if (!content) return;
                        const isMin = content.classList.contains('minimized');
                        if (!isMin) {
                            const rect = content.getBoundingClientRect();
                            content.dataset.prevLeft = content.style.left || `${rect.left}px`;
                            content.dataset.prevTop = content.style.top || `${rect.top}px`;
                            content.classList.add('minimized');
                            requestAnimationFrame(() => {
                                const r = content.getBoundingClientRect();
                                const margin = 12;
                                content.style.left = `${Math.max(margin, window.innerWidth - margin - r.width)}px`;
                                content.style.top = `${Math.max(margin, window.innerHeight - margin - r.height)}px`;
                            });
                        } else {
                            content.classList.remove('minimized');
                            const prevLeft = content.dataset.prevLeft || '';
                            const prevTop = content.dataset.prevTop || '';
                            if (prevLeft) content.style.left = prevLeft;
                            if (prevTop) content.style.top = prevTop;
                        }
                    }""",
                    show_progress=False,
                    queue=False
                )
                missing_model_btn.click(download_models, inputs=[state_topbar], outputs=[missing_model_modal, missing_model_list, missing_model_total_progress] + bar_buttons, api_name="download_models", show_progress=False)

                with gr.Row(elem_id='main_layout_row'):
                    with gr.Column(scale=2, visible=True, elem_classes='preview_column'):
                        with gr.Row():
                            progress_window = grh.Image(label='Preview', show_label=False, visible=True, height=768, elem_id='preview_generating',
                                                elem_classes=['main_view'], value="presets/welcome/welcome.png", interactive=False, show_download_button=False)
                            progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain', elem_id='finished_gallery',
                                                height=520, visible=False, elem_classes=['main_view', 'image_gallery'])
                            comparison_box = gr.HTML(visible=False, elem_id='comparison_box')
                            progress_video = gr.Video(label='Generated Video', show_label=True, visible=False, height=768, 
                                                elem_classes=['main_view', 'video_player'], elem_id='video_player', autoplay=True, show_share_button=False)
                            gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain', visible=False, height=768,
                                        elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                        elem_id='final_gallery', preview=True )

                        progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                            elem_id='progress-bar', elem_classes='progress-bar')

                        with gr.Row():
                            compare_btn = gr.Button("🖼️ Compare Input/Output", visible=False, size='sm')
                        with gr.Group(visible=False, elem_classes='infobox_group') as prompt_info_container:
                            prompt_info_box = gr.Markdown(toolbox.make_infobox_markdown(None, args_manager.args.theme), visible=False, elem_id='infobox', elem_classes='infobox')
                            prompt_info_close_btn = gr.Button(value='×', size='sm', elem_classes=['note_close_btn'], min_width=30, visible=False)

                        with gr.Group(visible=False, elem_classes='toolbox') as image_toolbox:
                            image_tools_box_title = gr.Markdown('<b>ToolBox</b>', visible=True)
                            prompt_info_button = gr.Button(value='ViewMeta', size='sm', visible=True)
                            prompt_regen_button = gr.Button(value='ReGenerate', size='sm', visible=True)
                            prompt_delete_button = gr.Button(value='DeleteImage', size='sm', visible=True)
                            prompt_info_button.click(toolbox.toggle_prompt_info, inputs=state_topbar, outputs=[prompt_info_box, prompt_info_close_btn, prompt_info_container, state_topbar], show_progress=False)
                            prompt_info_close_btn.click(toolbox.close_prompt_info, inputs=state_topbar, outputs=[prompt_info_box, prompt_info_close_btn, prompt_info_container, state_topbar], show_progress=False)

                        with gr.Group(visible=False, elem_classes='toolbox_note') as params_note_box:
                            params_note_info = gr.Markdown(elem_classes='note_info')
                            params_note_close_button = gr.Button(value='×', size='sm', elem_classes=['note_close_btn'], min_width=30)
                            params_note_input_name = gr.Textbox(show_label=False, placeholder="Type preset name here.", min_width=100, elem_classes='preset_input', visible=False)
                            params_note_delete_button = gr.Button(value='Enter', visible=False)
                            params_note_regen_button = gr.Button(value='Enter', visible=False)
                            params_note_preset_button = gr.Button(value='Enter', visible=False)

                        with gr.Accordion("Finished Images Catalog", open=False, visible=False, elem_id='finished_images_catalog') as index_radio:
                            gallery_index = gr.Radio(choices=None, label="Gallery_Index", value=None, show_label=False)
                    with gr.Column(scale=1, visible=False, elem_classes='scene_panel', elem_id='scene_panel') as scene_panel:
                        with gr.Row():
                            scene_additional_prompt = gr.Textbox(label="Blessing words", show_label=True, max_lines=1, elem_classes='scene_input')
                            scene_theme = gr.Radio(choices=modules.flags.scene_themes, label="Themes", value=modules.flags.scene_themes[0])

                        # Qwen Multiangle Camera Control
                        with gr.Accordion("📸 3D Camera Control", open=False, visible=False) as camera_control_accordion:
                            gr.HTML(value=qwen_multiangle.get_viewer_html(), elem_id="qwen_viewer_container")
                        
                        # Qwen Anglelight Lighting Control
                        with gr.Accordion("💡 3D Lighting Control", open=False, visible=False) as anglelight_control_accordion:
                            gr.HTML(value=qwen_anglelight.get_viewer_html(), elem_id="qwen_anglelight_viewer_container")

                            qwen_image_data = gr.Textbox(visible=False, elem_id="qwen_image_data")
                            qwen_image_data.change(
                                fn=None,
                                _js="""(val) => {
                                    const iframe = document.getElementById('qwen_multiangle_iframe');
                                    if (iframe && iframe.contentWindow) {
                                        iframe.contentWindow.postMessage({
                                            type: 'UPDATE_IMAGE',
                                            imageUrl: val
                                        }, '*');
                                    }
                                    const iframe2 = document.getElementById('qwen_anglelight_iframe');
                                    if (iframe2 && iframe2.contentWindow) {
                                        iframe2.contentWindow.postMessage({
                                            type: 'UPDATE_IMAGE',
                                            imageUrl: val
                                        }, '*');
                                    }
                                }""",
                                inputs=[qwen_image_data],
                                outputs=None
                            )

                        with gr.Accordion("🎞️ SAM3 Video Mask (Double Click to Open Frames Editor)", open=False, visible=False) as sam3_video_mask_accordion:
                            gr.HTML(value=sam3_video_mask.get_viewer_html(), elem_id="sam3_video_mask_html")
                            sam3_original_video_path = gr.State(None)
                            def sam3_translate_prompt_slim(prompt_text):
                                prompt_text = translator.normalize_prompt(prompt_text)
                                try:
                                    return translator.normalize_prompt(minicpm.translate(prompt_text, "Slim Model"))
                                except Exception:
                                    return translator.normalize_prompt(prompt_text)
                            with gr.Row():
                                sam3_input_video = gr.Video(label="Video (Upload)", show_label=True, source="upload", height=240, elem_id="sam3_input_video", show_share_button=False)
                                sam3_mask_video = gr.Video(label="Mask Video (Preview / Upload)", show_label=True, source="upload", height=240, elem_id="sam3_output_mask_video", show_share_button=False)
                            with gr.Accordion("💬 SAM3 Prompt Segmentation", open=False, visible=True):
                                with gr.Column():
                                    sam3_prompt_text = gr.Textbox(label="Segmentation Prompt", show_label=True, max_lines=1, placeholder="e.g. woman, dress", elem_id="sam3_prompt_text")
                                    sam3_trigger_translate_btn = gr.Button(visible=False, elem_id="sam3_trigger_translate_btn")
                                    sam3_trigger_translate_btn.click(fn=sam3_translate_prompt_slim, inputs=[sam3_prompt_text], outputs=[sam3_prompt_text], queue=False, show_progress=False)
                                    sam3_generate_btn = gr.Button("✅ Generate Mask", elem_id="sam3_generate_btn", size="sm")
                            sam3_editor_payload = gr.Textbox(visible=False, elem_id="sam3_editor_payload")
                            sam3_points_generate_btn = gr.Button("SAM3 Points Generate", visible=False, elem_id="sam3_points_generate_btn")
                            with gr.Accordion("🔧 SAM3 Params", open=False, visible=True):
                                with gr.Row():
                                    sam3_score_threshold_detection = gr.Slider(label="Detection Score Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                                    sam3_new_det_thresh = gr.Slider(label="New Detection Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
                                    sam3_fill_hole_area = gr.Slider(label="Fill Hole Area", minimum=0, maximum=512, step=1, value=16)
                                    sam3_recondition_every_nth_frame = gr.Slider(label="Recondition Every Nth Frame", minimum=1, maximum=128, step=1, value=16)
                                with gr.Row():
                                    sam3_postprocess_strength = gr.Slider(label="Mask Smoothing Strength", minimum=0, maximum=5, step=1, value=0)
                                    sam3_invert_mask = gr.Checkbox(label="Invert Mask", value=False)

                        with gr.Accordion("🎨 Style Selector", open=False, visible=False) as style_transfer_accordion:
                            gr.HTML(value=transfer_style_gallery.get_viewer_html(), elem_id="transfer_style_gallery_container_scene")

                        def check_camera_control_visibility(theme, state):
                            theme_l = theme.lower() if theme else ''
                            show_camera = bool(theme_l and 'multiangle' in theme_l)
                            show_light = bool(theme_l and ('anglelight' in theme_l or 'lightning' in theme_l))
                            show_style_transfer = bool(theme_l and 'flux2_styletransfer' in theme_l)
                            show_sam3 = bool(theme_l and 'sam3' in theme_l)
                            return (
                                gr.update(visible=show_camera, open=show_camera),
                                gr.update(visible=show_light, open=show_light),
                                gr.update(visible=show_style_transfer, open=False),
                                gr.update(visible=show_sam3, open=show_sam3),
                            )

                        scene_theme.change(
                            fn=check_camera_control_visibility,
                            inputs=[scene_theme, state_topbar],
                            outputs=[camera_control_accordion, anglelight_control_accordion, style_transfer_accordion, sam3_video_mask_accordion],
                            queue=False,
                            show_progress=False
                        )
                            
                        scene_canvas_image = grh.Image(label='Upload and canvas(1)', show_label=True, source='upload', type='numpy', tool='sketch', height=250, brush_color="#70FF81", mask_color=True, image_mode='RGBA', elem_id='scene_canvas')
                        scene_canvas_image_full = gr.State(None)
                        scene_canvas_image_backend = gr.State(None)
                        with gr.Row() as scene_input_images:
                            scene_input_image1 = grh.Image(label='Upload prompt image(2)', value=None, source='upload', type='numpy', image_mode='RGBA', show_label=True, height=300, show_download_button=False)
                            scene_input_image2 = grh.Image(label='Upload prompt image(3)', value=None, source='upload', type='numpy', image_mode='RGBA', show_label=True, height=300, show_download_button=False)
                            scene_input_image1_full = gr.State(None)
                            scene_input_image2_full = gr.State(None)
                        
                        def update_qwen_image(image):
                            if image is None:
                                return None
                            try:
                                if isinstance(image, np.ndarray):
                                    pil_image = Image.fromarray(image)
                                    max_size = 512
                                    if pil_image.width > max_size or pil_image.height > max_size:
                                        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                    
                                    buffered = io.BytesIO()
                                    pil_image.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    return f"data:image/png;base64,{img_str}"
                            except Exception as e:
                                print(f"Error converting image for Qwen viewer: {e}")
                                return None
                            return None

                        scene_input_image1.change(update_qwen_image, inputs=[scene_input_image1], outputs=[qwen_image_data], queue=False, show_progress=False)
                        
                        def on_video_upload(video_path):
                            if video_path is None:
                                return None, None, None
                            try:
                                preview_path = util.compress_video(video_path)
                                gr.Info("Compression completed!")
                                return preview_path, video_path, "scene"
                            except Exception as e:
                                gr.Warning(f"Compression failed: {e}")
                                return video_path, video_path, "scene"

                        scene_video = gr.Video(label="Video (Upload)", visible=False, source="upload", height=400)
                        scene_video.upload(on_video_upload, inputs=[scene_video], outputs=[scene_video, scene_original_video_path, active_video_source], show_progress=True)
                        scene_video_placeholder = gr.HTML('<div style="height: 400px; display: flex; align-items: center; justify-content: center; border: 2px dashed #ccc; border-radius: 8px; background: rgba(128,128,128,0.1); color: #888; font-size: 16px;"><span>Hide When Generating...</span></div>', visible=False, elem_id="scene_video_placeholder")
                        scene_audio = gr.Audio(label="Audio (Upload)", visible=False, source="upload", type="filepath")
                        scene_audio_placeholder = gr.HTML('<div style="padding: 20px; text-align: center; border: 2px dashed #ccc; border-radius: 8px; background: rgba(128,128,128,0.1); color: #888;">Hide When Generating...</div>', visible=False, elem_id="scene_audio_placeholder")
                        scene_additional_prompt_2 = gr.Textbox(label="Blessing words", show_label=True, max_lines=1, visible=False, elem_classes='scene_input_2', elem_id='scene_additional_prompt_2')
                        scene_var_number = gr.Slider(label='Duration(s)', minimum=0, maximum=60, step=1, value=3, visible=False)
                        
                        def sam3_on_video_upload_with_preview(video_path):
                            if video_path is None: return None, None, None
                            try: preview_path = util.compress_video(video_path); gr.Info("Compression completed!"); return preview_path, video_path, "sam3"
                            except Exception as e: gr.Warning(f"Compression failed: {e}"); return video_path, video_path, "sam3"

                        def sam3_cleanup_translator_and_vram():
                            try: translator.free_translator_model()
                            except Exception: pass
                            try:
                                import torch, gc
                                if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
                                gc.collect()
                            except Exception:
                                pass
                            try: model_management.soft_empty_cache()
                            except Exception: pass

                        def sam3_mask_opts(score_threshold_detection, new_det_thresh, fill_hole_area, recondition_every_nth_frame, postprocess_strength, invert_mask):
                            return dict(
                                precision="fp16",
                                score_threshold_detection=float(score_threshold_detection or 0.0),
                                new_det_thresh=float(new_det_thresh or 0.0),
                                det_nms_thresh=0.1,
                                fill_hole_area=int(fill_hole_area or 0),
                                recondition_every_nth_frame=int(recondition_every_nth_frame or 1),
                                image_size=1008,
                                postprocess_strength=int(postprocess_strength or 0),
                                postprocess_min_area=0,
                                debug_print=False,
                                invert_mask=bool(invert_mask),
                            )

                        def sam3_generate_mask_by_points(
                            original_video_path,
                            video_path,
                            editor_payload_json,
                            uploaded_mask_path,
                            score_threshold_detection,
                            new_det_thresh,
                            fill_hole_area,
                            recondition_every_nth_frame,
                            postprocess_strength,
                            invert_mask,
                        ):
                            effective_video_path = original_video_path or video_path
                            if effective_video_path is None: gr.Warning("Please upload a video first."); return uploaded_mask_path
                            if editor_payload_json is None or not str(editor_payload_json).strip():
                                if uploaded_mask_path: return uploaded_mask_path
                                gr.Warning("Please click the video and select targets in the popup, or upload a mask video directly."); return uploaded_mask_path
                            with worker.external_exclusive_task():
                                sam3_cleanup_translator_and_vram()
                                try:
                                    out_path = sam3_video_mask.run_sam3_video_mask(
                                        video_path=effective_video_path,
                                        editor_payload_json=editor_payload_json,
                                        **sam3_mask_opts(
                                            score_threshold_detection,
                                            new_det_thresh,
                                            fill_hole_area,
                                            recondition_every_nth_frame,
                                            postprocess_strength,
                                            invert_mask,
                                        ),
                                    )
                                    gr.Info("Mask generated!")
                                    return out_path
                                except Exception as e:
                                    logger.exception("SAM3 points mask generation failed")
                                    gr.Warning(f"SAM3 failed: {e}")
                                    unload_models_clicked(False)
                                    return uploaded_mask_path

                        def sam3_generate_mask_by_prompt(
                            original_video_path,
                            video_path,
                            prompt_text,
                            uploaded_mask_path,
                            score_threshold_detection,
                            new_det_thresh,
                            fill_hole_area,
                            recondition_every_nth_frame,
                            postprocess_strength,
                            invert_mask,
                        ):
                            effective_video_path = original_video_path or video_path
                            if effective_video_path is None: gr.Warning("Please upload a video first."); return uploaded_mask_path
                            if prompt_text is None or not str(prompt_text).strip():
                                if uploaded_mask_path: return uploaded_mask_path
                                gr.Warning("Please enter a prompt, or upload a mask video directly."); return uploaded_mask_path
                            with worker.external_exclusive_task():
                                try: prompt_text = translator.normalize_prompt(minicpm.translate(str(prompt_text), "Slim Model"))
                                except Exception: prompt_text = translator.normalize_prompt(str(prompt_text))
                                sam3_cleanup_translator_and_vram()
                                try:
                                    out_path = sam3_video_mask.run_sam3_video_mask_by_prompt(
                                        video_path=effective_video_path,
                                        prompt=str(prompt_text),
                                        **sam3_mask_opts(
                                            score_threshold_detection,
                                            new_det_thresh,
                                            fill_hole_area,
                                            recondition_every_nth_frame,
                                            postprocess_strength,
                                            invert_mask,
                                        ),
                                    )
                                    gr.Info("Mask generated!")
                                    return out_path
                                except Exception as e:
                                    logger.exception("SAM3 semantic prompt mask generation failed")
                                    gr.Warning(f"SAM3 failed: {e}")
                                    unload_models_clicked(False)
                                    return uploaded_mask_path

                        sam3_input_video.upload(sam3_on_video_upload_with_preview, inputs=[sam3_input_video], outputs=[sam3_input_video, sam3_original_video_path, active_video_source], show_progress=True)
                        sam3_points_generate_btn.click(sam3_generate_mask_by_points, inputs=[sam3_original_video_path, sam3_input_video, sam3_editor_payload, sam3_mask_video, sam3_score_threshold_detection, sam3_new_det_thresh, sam3_fill_hole_area, sam3_recondition_every_nth_frame, sam3_postprocess_strength, sam3_invert_mask], outputs=[sam3_mask_video], show_progress=True)
                        sam3_generate_btn.click(sam3_generate_mask_by_prompt, inputs=[sam3_original_video_path, sam3_input_video, sam3_prompt_text, sam3_mask_video, sam3_score_threshold_detection, sam3_new_det_thresh, sam3_fill_hole_area, sam3_recondition_every_nth_frame, sam3_postprocess_strength, sam3_invert_mask], outputs=[sam3_mask_video], show_progress=True)
                        with gr.Accordion("🔧 Advanced Parameters", open=False, visible=True):
                            scene_var_number2 = gr.Slider(label='Int Value 2', minimum=0, maximum=60, step=1, value=1, visible=False)
                            scene_var_number3 = gr.Slider(label='Float Value 1', minimum=0.0, maximum=1.0, step=0.05, value=0.0, visible=False)
                            scene_var_number4 = gr.Slider(label='Float Value 2', minimum=0.0, maximum=1.0, step=0.05, value=0.0, visible=False)
                            scene_var_number5 = gr.Slider(label='Float Value 3', minimum=0.0, maximum=1.0, step=0.05, value=0.0, visible=False)
                            scene_var_number6 = gr.Slider(label='Float Value 4', minimum=0.0, maximum=1.0, step=0.05, value=0.0, visible=False)
                            with gr.Row():
                                scene_var_number7 = gr.Slider(label='Int Value 3', minimum=0, maximum=60, step=1, value=0, visible=False, scale=1)
                                scene_var_number8 = gr.Slider(label='Int Value 4', minimum=0, maximum=60, step=1, value=0, visible=False, scale=1)
                            with gr.Row():
                                scene_var_number9 = gr.Slider(label='Int Value 5', minimum=0, maximum=60, step=1, value=0, visible=False, scale=1)
                                scene_var_number10 = gr.Slider(label='Int Value 6', minimum=0, maximum=60, step=1, value=0, visible=False, scale=1)
                            scene_steps = gr.Slider(label='Scene Steps', minimum=1, maximum=30, step=1, value=20, visible=False)
                            with gr.Row():
                                scene_switch_option1 = gr.Checkbox(label='Switch Option 1', value=False, visible=False)
                                scene_switch_option2 = gr.Checkbox(label='Switch Option 2', value=False, visible=False)
                            with gr.Row():
                                scene_switch_option3 = gr.Checkbox(label='Switch Option 3', value=False, visible=False)
                                scene_switch_option4 = gr.Checkbox(label='Switch Option 4', value=False, visible=False)
                            with gr.Row():
                                scene_aspect_ratio = gr.Radio(choices=modules.flags.scene_aspect_ratios[:3], label="Aspect Ratios", value=modules.flags.scene_aspect_ratios[0], elem_classes=['scene_aspect_ratio_selections'])
                            with gr.Row():
                                scene_prompt_preset_button = gr.Button(value='Save the current parameters as a preset package')
                        with gr.Row():
                            scene_image_number = gr.Slider(label='Image Number', minimum=1, maximum=5, step=1, value=1)
                            scene_mask_color = gr.ColorPicker(label="Scene brush color", value="#70FF81", elem_id="scene_brush_color")

                        model_filter_state = gr.State(True)
                        model_filter_sync_lock = gr.State(False)
                        scene_to_main_sync_lock = gr.State(False)

                        with gr.Accordion("⚙️ Scene Model Selections", open=False, visible=True, elem_id="scene_model_selections") as scene_model_selections:
                            with gr.Row():
                                scene_base_model = gr.Dropdown(
                                    label='Base Model (or HighNoise)',
                                    choices=modules.config.model_filenames,
                                    value=modules.config.default_base_model_name,
                                    show_label=True,
                                    elem_id="scene_model_dropdown_base",
                                    elem_classes="model-dropdown",
                                    interactive=True,
                                )
                                scene_refiner_model = gr.Dropdown(
                                    label='Refiner (or LowNoise)',
                                    choices=['None'] + modules.config.get_base_model_list('Fooocus', None),
                                    value=modules.config.default_refiner_model_name,
                                    show_label=True,
                                    elem_id="scene_model_dropdown_refiner",
                                    elem_classes="model-dropdown",
                                    interactive=True,
                                    visible=False,
                                )
                            with gr.Row():
                                scene_use_lora = gr.Checkbox(label='Use LoRAs', value=True, visible=True)
                            lora_group = gr.Group(visible=True)
                            with lora_group:
                                from modules.lora_trigger_manager import get_lora_trigger_word, update_trigger_word, save_trigger_word, send_trigger_to_prompt

                                scene_lora_trigger_words = []
                                scene_lora_send_to_prompt_btns = []
                                scene_lora_save_btns = []
                                with gr.Row():
                                    show_trigger_words_panel = gr.Checkbox(label='Show Trigger Words Panel', value=False, elem_classes='show_trigger_words_panel')
                                    scene_use_model_filter_checkbox = gr.Checkbox(label='Use Model Filters', value=True, elem_classes='use_model_filter_checkbox')
                                trigger_word_containers = []
                                with gr.Row():
                                    scene_lora_model = gr.Dropdown(label='LoRA 1 / HighNoise ',
                                                                  choices=['None'] + modules.config.lora_filenames, value='None', 
                                                                  elem_classes='lora_model', scale=5, elem_id="scene_lora_dropdown_0",interactive=True)
                                    scene_lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight, 
                                                                 maximum=modules.config.default_loras_max_weight, step=0.05, value=1.0,
                                                                 elem_classes='lora_weight', scale=5,interactive=True)\

                                with gr.Row(visible=False) as trigger_container_0:
                                    trigger_word_containers.append(trigger_container_0)
                                    trigger_word_value_0 = get_lora_trigger_word(scene_lora_model.value) if scene_lora_model.value != 'None' else ''
                                    scene_lora_trigger_word_0 = gr.Textbox(label='Trigger Word', value=trigger_word_value_0,
                                                                       placeholder='Input LoRA trigger word',
                                                                       elem_id="scene_lora_trigger_word_0", min_width=300, lines=1, scale=5)
                                    scene_lora_trigger_words.append(scene_lora_trigger_word_0)
                                    with gr.Column(min_width=80):
                                        scene_send_to_prompt_btn_0 = gr.Button("✅", variant="secondary",elem_id=f"scene_lora_send_to_prompt_0")
                                        scene_lora_save_btns.append(gr.Button("💾", variant="secondary",elem_id=f"scene_lora_save_0"))
                                    scene_lora_save_btns[0].click(
                                        fn=save_trigger_word,
                                        inputs=[scene_lora_model, scene_lora_trigger_word_0],
                                        outputs=[scene_lora_trigger_word_0]
                                    )
                                    scene_lora_send_to_prompt_btns.append(scene_send_to_prompt_btn_0)
                                    scene_send_to_prompt_btn_0.click(
                                        fn=send_trigger_to_prompt,
                                        inputs=[scene_lora_model, scene_lora_trigger_word_0],
                                        outputs=[scene_lora_send_to_prompt_btns[0]]
                                    )
                                with gr.Row():
                                    scene_lora_model_2 = gr.Dropdown(label='LoRA 2 / HighNoise',
                                                                   choices=['None'] + modules.config.lora_filenames, value='None', 
                                                                   elem_classes='lora_model', scale=5, elem_id="scene_lora_dropdown_1",interactive=True)
                                    scene_lora_weight_2 = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight, 
                                                                    maximum=modules.config.default_loras_max_weight, step=0.05, value=1.0,
                                                                    elem_classes='lora_weight', scale=5,interactive=True)

                                with gr.Row(visible=False) as trigger_container_1:
                                    trigger_word_containers.append(trigger_container_1)
                                    trigger_word_value_1 = get_lora_trigger_word(scene_lora_model.value) if scene_lora_model.value != 'None' else ''
                                    scene_lora_trigger_word_1 = gr.Textbox(label='Trigger Word', value=trigger_word_value_0,
                                                                       placeholder='Input LoRA trigger word',
                                                                       elem_id="scene_lora_trigger_word_0", min_width=300, lines=1, scale=5)
                                    scene_lora_trigger_words.append(scene_lora_trigger_word_1)
                                    with gr.Column(min_width=80):
                                        scene_send_to_prompt_btn_1 = gr.Button("✅", variant="secondary",elem_id=f"scene_lora_send_to_prompt_0")
                                        scene_lora_save_btns.append(gr.Button("💾", variant="secondary",elem_id=f"scene_lora_save_0"))
                                    scene_lora_save_btns[0].click(
                                        fn=save_trigger_word,
                                        inputs=[scene_lora_model, scene_lora_trigger_word_1],
                                        outputs=[scene_lora_trigger_word_1]
                                    )
                                    scene_lora_send_to_prompt_btns.append(scene_send_to_prompt_btn_1)
                                    scene_send_to_prompt_btn_1.click(
                                        fn=send_trigger_to_prompt,
                                        inputs=[scene_lora_model, scene_lora_trigger_word_1],
                                        outputs=[scene_lora_send_to_prompt_btns[1]]
                                    )
                                with gr.Row():
                                    scene_lora_model_3 = gr.Dropdown(label='LoRA 3 / LowNoise',
                                                                   choices=['None'] + modules.config.lora_filenames, value='None', 
                                                                   elem_classes='lora_model', scale=5, elem_id="scene_lora_dropdown_2",interactive=True)
                                    scene_lora_weight_3 = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight, 
                                                                    maximum=modules.config.default_loras_max_weight, step=0.05, value=1.0,
                                                                    elem_classes='lora_weight', scale=5,interactive=True)

                                with gr.Row(visible=False) as trigger_container_2:
                                    trigger_word_containers.append(trigger_container_2)
                                    trigger_word_value_2 = get_lora_trigger_word(scene_lora_model_3.value) if scene_lora_model_3.value != 'None' else ''
                                    scene_lora_trigger_word_2 = gr.Textbox(label='Trigger Word', value=trigger_word_value_2,
                                                                       placeholder='Input LoRA trigger word',
                                                                       elem_id="scene_lora_trigger_word_2", min_width=300, lines=1, scale=5)
                                    scene_lora_trigger_words.append(scene_lora_trigger_word_2)
                                    with gr.Column(min_width=80):
                                        scene_send_to_prompt_btn_2 = gr.Button("✅", variant="secondary",elem_id=f"scene_lora_send_to_prompt_2")
                                        scene_lora_save_btns.append(gr.Button("💾", variant="secondary",elem_id=f"scene_lora_save_2"))
                                    scene_lora_save_btns[2].click(
                                        fn=save_trigger_word,
                                        inputs=[scene_lora_model_3, scene_lora_trigger_word_2],
                                        outputs=[scene_lora_trigger_word_2]
                                    )
                                    scene_lora_send_to_prompt_btns.append(scene_send_to_prompt_btn_2)
                                    scene_send_to_prompt_btn_2.click(
                                        fn=send_trigger_to_prompt,
                                        inputs=[scene_lora_model_3, scene_lora_trigger_word_2],
                                        outputs=[scene_lora_send_to_prompt_btns[2]]
                                    )
                                with gr.Row():
                                    scene_lora_model_4 = gr.Dropdown(label='LoRA 4 / LowNoise',
                                                                   choices=['None'] + modules.config.lora_filenames, value='None', 
                                                                   elem_classes='lora_model', scale=5, elem_id="scene_lora_dropdown_3",interactive=True)
                                    scene_lora_weight_4 = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight, 
                                                                    maximum=modules.config.default_loras_max_weight, step=0.05, value=1.0,
                                                                    elem_classes='lora_weight', scale=5,interactive=True)

                                with gr.Row(visible=False) as trigger_container_3:
                                    trigger_word_containers.append(trigger_container_3)
                                    trigger_word_value_3 = get_lora_trigger_word(scene_lora_model_4.value) if scene_lora_model_4.value != 'None' else ''
                                    scene_lora_trigger_word_3 = gr.Textbox(label='Trigger Word', value=trigger_word_value_3,
                                                                       placeholder='Input LoRA trigger word',
                                                                       elem_id="scene_lora_trigger_word_3", min_width=300, lines=1, scale=5)
                                    scene_lora_trigger_words.append(scene_lora_trigger_word_3)
                                    with gr.Column(min_width=80):
                                        scene_send_to_prompt_btn_3 = gr.Button("✅", variant="secondary",elem_id=f"scene_lora_send_to_prompt_3")
                                        scene_lora_save_btns.append(gr.Button("💾", variant="secondary",elem_id=f"scene_lora_save_3"))
                                    scene_lora_save_btns[3].click(
                                        fn=save_trigger_word,
                                        inputs=[scene_lora_model_4, scene_lora_trigger_word_3],
                                        outputs=[scene_lora_trigger_word_3]
                                    )
                                    scene_lora_send_to_prompt_btns.append(scene_send_to_prompt_btn_3)
                                    scene_send_to_prompt_btn_3.click(
                                        fn=send_trigger_to_prompt,
                                        inputs=[scene_lora_model_4, scene_lora_trigger_word_3],
                                        outputs=[scene_lora_send_to_prompt_btns[3]]
                                    )
                                show_trigger_words_panel.change(
                                    fn=lambda visible: [gr.update(visible=visible)] * len(trigger_word_containers),
                                    inputs=[show_trigger_words_panel],
                                    outputs=trigger_word_containers,
                                    queue=False, show_progress=False
                                )
                            with gr.Row():
                                scene_refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')

                                scene_lora_ctrls = [scene_lora_model, scene_lora_weight,
                                                    scene_lora_model_2, scene_lora_weight_2,
                                                    scene_lora_model_3, scene_lora_weight_3,
                                                    scene_lora_model_4, scene_lora_weight_4]
                            scene_use_lora.change(
                                fn=lambda x: gr.update(visible= x),
                                inputs=scene_use_lora,
                                outputs=lora_group
                            )
                        with gr.Row():
                            scene_seed_random = gr.Checkbox(label='Random', value=True)
                            scene_image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False)
                        scene_mask_color.change(lambda x: gr.update(brush_color=x),inputs=scene_mask_color,
                            outputs=scene_canvas_image,
                            queue=False,show_progress=False)
                        
                with gr.Group(visible=False, elem_classes='identity_note') as identity_dialog:
                    with gr.Tabs():
                        with gr.Tab(label='IdentityCard') as bind_id_tab:
                            with gr.Row():
                                with gr.Column(scale=5, min_width=250):
                                    current_id_info = gr.Markdown(elem_classes='note_info')
                                with gr.Column(scale=1, min_width=50):
                                    current_upstream_status = gr.Markdown(elem_classes='note_info')
                                    identity_export_btn = gr.Button(value='Export identity', size='sm', min_width=35, elem_classes='identity_export', visible=False)
                            with gr.Row(visible=True) as input_identity:
                                with gr.Column(scale=4, min_width=126):
                                    input_qr_title = gr.Markdown(elem_classes='input_note_info', value='<b>Upload QrCode to bind</b>')
                                    identity_qr = grh.Image(label='Identity QrCode', source='upload', type='numpy', height=126, width=126, elem_classes='identity_qr')
                                with gr.Column(scale=4, min_width=150):
                                    input_id_title = gr.Markdown(elem_classes='input_note_info', value='<b>Input identity to bind</b>')
                                    identity_nick_input = gr.Textbox(show_label=False, max_lines=1, container=False, placeholder="Type nickname here.", min_width=50, elem_classes='identity_input2')
                                    with gr.Row():
                                        with gr.Column(scale=2, min_width=20):
                                            identity_areacode = gr.Dropdown(choices=modules.flags.areacode, value='86-CN-中国', container=False, min_width=20, elem_id='areacode',elem_classes='identity_input3')
                                        with gr.Column(scale=3, min_width=30):
                                            identity_tele_input = gr.Textbox(show_label=False, max_lines=1, container=False, placeholder="Type telephone here.", min_width=30, elem_classes='identity_input2')
                                    identity_bind_button = gr.Button(value='Bind identity', min_width=40, visible=True)
                            with gr.Row(visible=False) as input_id_display:
                                input_id_info = gr.Markdown(elem_classes='input_id_info', value='input identity', min_width=200, visible=True)
                                identity_change_button = gr.Button(value='Change identity', min_width=40, visible=True)
                            with gr.Row():
                                identity_vcode_input = gr.Textbox(show_label=False, max_lines=1, container=False, visible=False, placeholder="Type Verification here.", min_width=70, elem_classes='identity_input')
                                identity_verify_button = gr.Button(value='Verify identity', elem_classes='identity_button', visible=False)
                            with gr.Row():
                                identity_phrase_input = gr.Textbox(show_label=False, type='password', visible=False, container=False, placeholder="Type ID phrases here.", min_width=150, elem_classes='identity_input')
                                identity_phrases_set_button = gr.Button(value='Setting ID phrases', elem_classes='identity_button', visible=False)
                                identity_phrases_confirm_button = gr.Button(value='Confirm ID phrases', elem_classes='identity_button', visible=False)
                                identity_confirm_button = gr.Button(value='Confirm identity', elem_classes='identity_button', visible=False)
                                identity_unbind_button = gr.Button(value='Unbind identity', min_width=35, elem_classes='identity_button', visible=False)
                    identity_note_info = gr.Markdown(elem_classes='note_info', value=simpleai.identity_note)
                
                    identity_input = [identity_nick_input, identity_areacode, identity_tele_input, identity_qr]
                    identity_input_info = [input_id_info, state_topbar]
                    identity_ctrls = [identity_note_info, input_identity, input_id_display, identity_vcode_input, identity_verify_button, identity_phrase_input, identity_phrases_set_button, identity_phrases_confirm_button, identity_confirm_button, identity_unbind_button]
                    identity_bind_button.click(simpleai.bind_identity, inputs=[identity_nick_input, identity_areacode, identity_tele_input], outputs=identity_ctrls + [input_id_info], show_progress=False)
                    identity_change_button.click(simpleai.change_identity,  outputs=identity_ctrls + identity_input, show_progress=False)
                    identity_verify_button.click(simpleai.verify_identity, inputs=identity_input_info + [identity_vcode_input], outputs=identity_ctrls, show_progress=False)
                    identity_phrases_set_button.click(lambda a, b, c: simpleai.set_phrases(a,b,c,'set'), inputs=identity_input_info + [identity_phrase_input], outputs=identity_ctrls + [current_id_info], show_progress=False)
                    identity_export_btn.click(topbar.export_identity, inputs=state_topbar, outputs=system_params, show_progress=False) \
                        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}') \
                        .then(fn=lambda x: '' if 'user_qr' not in x else x.pop('user_qr'), inputs=state_topbar,  show_progress=False)

                    identity_qr.upload(simpleai.trigger_input_identity, inputs=identity_qr, outputs=identity_ctrls + [input_id_info], show_progress=False, queue=False)
                
                nav_bars = [bar_store_button] + bar_buttons
                bar_store_button.click(topbar.toggle_preset_store, inputs=state_topbar, outputs=[preset_store, preset_store_list, system_params, identity_dialog, current_id_info, current_upstream_status, identity_export_btn] + identity_ctrls + identity_input, show_progress=False).then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')
                preset_store_list.click(topbar.update_navbar_from_mystore, inputs=[preset_store_list, state_topbar], outputs=nav_bars + [system_params], show_progress=False).then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')
                
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=12):
                        with gr.Group(elem_classes="prompt-container"):
                            prompt = gr.Textbox(
                                show_label=False, placeholder="Type prompt here or paste parameters.",
                                elem_id="positive_prompt", container=False, autofocus=False, lines=4
                            )
                            clear_prompt_btn = gr.Button(value="x", elem_classes=["clear-prompt-btn"], visible=True)
                            prompt_token_counter = gr.HTML(visible=True, value=0, elem_classes=["tokenCounter"], elem_id="token_counter")
                            tag_helper_btn = gr.HTML('<i class="fa-solid fa-tags"></i>', elem_classes=["tagHelper"], elem_id="tag_helper_btn")

                        def calculateTokenCounter(text, style_selections):
                            if len(text) < 1:
                                return 0
                            num=topbar.prompt_token_prediction(text, style_selections)
                            return str(num)

                        clear_prompt_btn.click(fn=lambda: "", outputs=prompt, queue=False, show_progress=False)
                        default_prompt = modules.config.default_prompt
                        if isinstance(default_prompt, str) and default_prompt != '':
                            shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                    with gr.Column(scale=2, min_width=40) as prompt_internal_panel:
                        random_button = gr.Button(value="RandomPrompt", elem_id="random_prompt_button", elem_classes='type_row_half', size="sm", min_width = 70)
                        super_prompter = gr.Button(value="SuperPrompt", interactive=False, elem_id="super_prompter_button", elem_classes='type_row_half', size="sm", min_width = 70)
                    with gr.Column(scale=2, min_width=40):
                        generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True, min_width = 70)
                        load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False, min_width = 70)
                        skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False, min_width = 70)
                        stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False, min_width = 70)

                        def stop_clicked(currentTask):
                            currentTask.last_stop = 'stop'
                            if (currentTask.processing):
                                worker.worker.interrupt_processing()
                            return currentTask

                        def skip_clicked(currentTask):
                            currentTask.last_stop = 'skip'
                            if (currentTask.processing):
                                worker.worker.interrupt_processing()
                            return currentTask

                        stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                        skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

                with gr.Accordion(label='Parallel Translation', visible=True, open=False, elem_id='translation_preview_accordion', elem_classes='translation_preview_accordion') as translation_preview:
                    translated_prompt = gr.HTML(value="", elem_classes='translation-preview')
                    translation_preview_open = gr.Checkbox(value=False, elem_id="translation_preview_open",visible=False,container=False)
                    def translate_prompt(text):
                        from modules.util import is_chinese
                        try:
                            if not text.strip():
                                return ""
                            translation_method = ads.get_admin_default('translation_methods')
                            is_generating = check_generating_state()
                            if not is_generating and translation_method == 'Big Model' and MiniCPM.get_enable():
                                if is_chinese(text):
                                    return minicpm.translate(text)
                                else:
                                    return minicpm.translate_cn(text)
                            if is_generating :
                                logger.info("Translation disable MiniCPM while generating.")
                            return translator.toggle(text, translation_method)
                        except Exception as e:
                            return f"Translation error：{str(e)}"

                    def handle_translation_preview_open(current_prompt, is_open):
                        if is_open:
                            return translate_prompt(current_prompt)
                        return translated_prompt.value
                    trigger_translation_btn = gr.Button(visible=False, elem_id="trigger_translation_btn")
                    trigger_translation_btn.click(fn=handle_translation_preview_open,inputs=[prompt, translation_preview_open],outputs=[translated_prompt])

                state_prompt_history = gr.State([])
                with gr.Accordion(label='Prompt History', visible=False, open=True) as prompt_history:
                    history_prompts = gr.Dataset(components=[prompt],label='Click to reuse:',samples=[[p] for p in state_prompt_history.value[-5:]],type='index')
                history_prompts.click(lambda x, y: y[x] if 0 <= x < len(y) else "",
                                      inputs=[history_prompts, state_prompt_history],
                                      outputs=prompt,show_progress=False,queue=False)

                with gr.Accordion(label='Wildcards & Batch Prompts', visible=False, open=True) as prompt_wildcards:
                    gr.HTML(value='<a href="wildcards/readme" target="_blank" rel="noopener noreferrer">Wildcards readme</a>')
                    with gr.Accordion(label="🎯 Wildcards Helper", visible=True, open=False):
                        wildcard_names = [x[0] for x in wildcards.get_wildcards_samples(trans=False)]
                        with gr.Row():
                            with gr.Column(scale=1, min_width=220):
                                wc_target = gr.Dropdown(label="Target",value="Array (batch)",choices=["Array (batch)", "Single in prompt"])
                            with gr.Column(scale=1, min_width=220):
                                wc_method = gr.Dropdown(label="Method",value="Random Select",choices=["Random Select", "In order"])
                            with gr.Column(scale=1, min_width=220):
                                wc_seed_mode = gr.Dropdown(label="Seed mode",value="Fixed seed",choices=["Fixed seed", "Random seed"])
                        with gr.Row():
                            with gr.Column(scale=1, min_width=220):
                                wc_name = gr.Dropdown(label="Wildcard",value=wildcard_names[0] if len(wildcard_names) > 0 else "",choices=wildcard_names)
                            with gr.Column(scale=1, min_width=220):
                                wc_count = gr.Number(label="Count", value=1, precision=0, minimum=1, step=1)
                            with gr.Column(scale=1, min_width=220):
                                wc_start = gr.Number(label="Start index", value=1, precision=0, minimum=1, step=1, visible=False)
                                wc_group_size = gr.Number(label="Group size", value=1, precision=0, minimum=1, step=1, visible=True)
                        wc_preview = gr.HTML(value="")
                        with gr.Row():
                            with gr.Column(scale=5, min_width=220):
                                wc_insert_btn = gr.Button(value="Append to prompt")
                            with gr.Column(scale=5, min_width=220):
                                wc_manage_personal_btn = gr.Button(value="Wildcards Editor")

                        wc_target.change(wildcards.update_wildcards_helper_controls, inputs=[wc_target, wc_method, wc_seed_mode, wc_name, wc_count, wc_start, wc_group_size], outputs=[wc_start, wc_group_size], show_progress=False, queue=False)
                        wc_method.change(wildcards.update_wildcards_helper_controls, inputs=[wc_target, wc_method, wc_seed_mode, wc_name, wc_count, wc_start, wc_group_size], outputs=[wc_start, wc_group_size], show_progress=False, queue=False)

                        for c in [wc_target, wc_method, wc_seed_mode, wc_name, wc_count, wc_start, wc_group_size]:
                            c.change(wildcards.update_wildcards_helper_preview, inputs=[wc_target, wc_method, wc_seed_mode, wc_name, wc_count, wc_start, wc_group_size], outputs=[wc_preview], show_progress=False, queue=False)

                        wc_insert_btn.click(
                            wildcards.append_wildcards_helper_tag_to_prompt,
                            inputs=[prompt, wc_target, wc_method, wc_seed_mode, wc_name, wc_count, wc_start, wc_group_size],
                            outputs=[prompt],
                            show_progress=False,
                            queue=False
                        )
                    wildcards_list = gr.Dataset(components=[prompt], type='index', label='Wildcards examples: [__color__:L3:4] = 3 items in order starting from the 4th. [__color__:3] = 3 random candidates (3 images). __color__ = 1 random per image.', samples=wildcards.get_wildcards_samples(), visible=True, samples_per_page=28)
                    with gr.Accordion(label='Words/phrases of wildcard', visible=True, open=False) as words_in_wildcard:
                        wildcard_tag_name_selection = gr.Dataset(components=[prompt], label='Words:', samples=wildcards.get_words_of_wildcard_samples(), visible=True, samples_per_page=30, type='index')
                    wildcards_list.click(wildcards.add_wildcards_and_array_to_prompt, inputs=[wildcards_list, prompt, state_topbar], outputs=[prompt, wildcard_tag_name_selection, words_in_wildcard], show_progress=False, queue=False)
                    wildcard_tag_name_selection.click(wildcards.add_word_to_prompt, inputs=[wildcards_list, wildcard_tag_name_selection, prompt, state_topbar], outputs=prompt, show_progress=False, queue=False)
                    wildcards_array = [prompt_wildcards, words_in_wildcard, wildcards_list, wildcard_tag_name_selection, wc_name]

                    def wildcards_array_show(state_params):
                        user_did = state_params["user"].get_did() if isinstance(state_params, dict) and "user" in state_params and state_params["user"] is not None else None
                        wildcard_in = state_params.get("wildcard_in_wildcards", "root") if isinstance(state_params, dict) else "root"
                        names = [x[0] for x in wildcards.get_wildcards_samples(trans=False, user_did=user_did)]
                        name_value = names[0] if len(names) > 0 else ""
                        return (
                            [gr.update(visible=True)] * 2
                            + [
                                gr.Dataset.update(visible=True, samples=wildcards.get_wildcards_samples(user_did=user_did)),
                                gr.Dataset.update(visible=True, samples=wildcards.get_words_of_wildcard_samples(wildcard_in, user_did=user_did)),
                                gr.update(choices=names, value=name_value),
                            ]
                        )

                    def wildcards_array_hidden():
                        return (
                            [gr.update(visible=False)] * 2
                            + [
                                gr.Dataset.update(visible=False, samples=[]),
                                gr.Dataset.update(visible=False, samples=[]),
                                gr.update(),
                            ]
                        )

                    wildcards_array_hold = [gr.update()] * 5

                    user_personal_wildcards_modal = gr.Box(visible=False, elem_id="user_personal_wildcards_modal", elem_classes=["modal", "user-wildcards-modal"])
                    with user_personal_wildcards_modal:
                        user_personal_wildcards_modal_content = gr.Column(elem_classes=["modal-content"], elem_id="user_personal_wildcards_modal_content", scale=1, min_width=800)
                        with user_personal_wildcards_modal_content:
                            user_personal_wildcards_title = gr.Markdown("### Personal Wildcards", elem_id="user_personal_wildcards_modal_handle")
                            user_personal_wildcards_status = gr.Markdown("")
                            with gr.Row():
                                user_personal_wildcards_select = gr.Dropdown(label="File", choices=[], value=None, scale=5)
                                user_personal_wildcards_refresh_btn = gr.Button(value="🔄 Refresh", size="sm", min_width=60, scale=1)
                                user_personal_wildcards_close_btn = gr.Button(value="❌ Close", size="sm", min_width=60, scale=1)
                            with gr.Row():
                                user_personal_wildcards_name = gr.Textbox(label="Name", placeholder="e.g. my_style (saved as .txt)", lines=1)
                            with gr.Row():
                                user_personal_wildcards_content = gr.Textbox(label="Content (one per line)", lines=12, elem_id="user_personal_wildcards_content", elem_classes=["line-overlay-textbox"])
                            with gr.Row():
                                user_personal_wildcards_save_btn = gr.Button(value="💾 Save", interactive=False)
                                user_personal_wildcards_delete_btn = gr.Button(value="🗑️ Delete", variant="secondary", interactive=False)
                            with gr.Accordion(label="📦 Upload .txt", open=False):
                                user_personal_wildcards_upload_file = gr.File(label="Choose a .txt file", file_types=[".txt"], type="file")
                                user_personal_wildcards_upload_name = gr.Textbox(label="Save as (optional)", placeholder="Leave blank to use original filename", lines=1)
                                user_personal_wildcards_upload_btn = gr.Button(value="Upload/Overwrite")

                    wc_manage_personal_btn_outputs = [user_personal_wildcards_modal, user_personal_wildcards_select, user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_status, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn]
                    def _deny_personal_wildcards_open(msg=None):
                        if msg:
                            try:
                                gr.Info(msg)
                            except Exception:
                                pass
                        return (gr.update(visible=False),) + tuple(gr.update() for _ in range(6))

                    def _open_personal_wildcards_guard(state_params):
                        try:
                            user = state_params.get("user", None) if isinstance(state_params, dict) else None
                            user_did = user.get_did() if user is not None and hasattr(user, "get_did") else None
                            if user_did is None or shared.token.is_guest(user_did):
                                lang = state_params.get("__lang", "cn") if isinstance(state_params, dict) else "cn"
                                msg = "请先登入身份。" if lang == "cn" else "Please sign in."
                                return _deny_personal_wildcards_open(msg)
                        except Exception:
                            return _deny_personal_wildcards_open()
                        return wildcards.personal_wildcards_open(state_params)

                    wc_manage_personal_btn.click(fn=_open_personal_wildcards_guard, inputs=[state_topbar], outputs=wc_manage_personal_btn_outputs, show_progress=False, queue=True)
                    user_personal_wildcards_close_btn.click(fn=wildcards.personal_wildcards_close, outputs=[user_personal_wildcards_modal], show_progress=False, queue=False)
                    user_personal_wildcards_refresh_btn.click(fn=wildcards.personal_wildcards_refresh, inputs=[state_topbar, user_personal_wildcards_select], outputs=[user_personal_wildcards_select, user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_status, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn], show_progress=False, queue=False)
                    user_personal_wildcards_select.change(fn=wildcards.personal_wildcards_load, inputs=[state_topbar, user_personal_wildcards_select], outputs=[user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_status, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn], show_progress=False, queue=False)
                    user_personal_wildcards_name.change(fn=wildcards.personal_wildcards_update_actions, inputs=[user_personal_wildcards_name], outputs=[user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn], show_progress=False, queue=False)
                    user_personal_wildcards_save_btn.click(fn=wildcards.personal_wildcards_save, inputs=[state_topbar, user_personal_wildcards_name, user_personal_wildcards_content], outputs=[user_personal_wildcards_status, user_personal_wildcards_select, user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn, wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False)
                    user_personal_wildcards_delete_btn.click(fn=wildcards.personal_wildcards_delete, inputs=[state_topbar, user_personal_wildcards_name], outputs=[user_personal_wildcards_status, user_personal_wildcards_select, user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn, wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False)
                    user_personal_wildcards_upload_btn.click(fn=wildcards.personal_wildcards_upload, inputs=[state_topbar, user_personal_wildcards_upload_file, user_personal_wildcards_upload_name], outputs=[user_personal_wildcards_select, user_personal_wildcards_name, user_personal_wildcards_content, user_personal_wildcards_status, user_personal_wildcards_save_btn, user_personal_wildcards_delete_btn, wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False)
            
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                prompt_panel_checkbox = gr.Checkbox(label='Prompt Panel', value=False, container=False, elem_classes='min_check')
                qwen_tts_checkbox = gr.Checkbox(label='TTS Audio', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced+', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            
            engine_class_display = gr.HTML(visible=False, value="Z-image", elem_classes=["engineClass"], elem_id='engine_class')
            with gr.Row(visible=False, elem_id="tts_panel") as tts_panel:
                with gr.Column():
                    qwen_send_target_options = {
                        "Voice Clone / Reference Audio": "qwen_clone_ref_audio",
                        "Dialogue / Role 1 Reference Audio": "qwen_role_1_audio",
                        "Dialogue / Role 2 Reference Audio": "qwen_role_2_audio",
                        "Dialogue / Role 3 Reference Audio": "qwen_role_3_audio",
                        "Dialogue / Role 4 Reference Audio": "qwen_role_4_audio",
                        "Scene / Audio (Upload)": "scene_audio",
                    }
                    qwen_send_target_choices = list(qwen_send_target_options.keys())
                    with gr.Tabs():
                        with gr.Tab("Voice Design"):
                            qwen_design_text = gr.Textbox(label="Text to Speech", lines=3, placeholder="Enter text here...[pause=800ms] or [pause=0.8s] can add pause between sentences.")
                            qwen_tts_style_presets = {
                                "Catgirl (Neko)": "Cute catgirl voice: high-pitched, bright and sweet, youthful and playful. Add occasional short interjections like 'nya', 'meow', 'na', 'ne', 'ya' (not every sentence). Expressive with subtle emotional shifts: shy -> softer, breathy, slightly shaky; tsundere -> quick pitch rise and a small 'hmph'; teary -> light sob or choked tone. Optionally add close-mic ASMR details (soft breathing, whispery delivery) while keeping articulation clear.",
                                "Warm Female": "Female, mid-20s, warm and friendly, medium pace, clear articulation, slight smile in voice, natural breath and gentle intonation.",
                                "News Anchor": "Male, 30s, calm professional news anchor, steady rhythm, neutral emotion, crisp consonants, confident delivery, minimal pitch fluctuation.",
                                "Energetic Teen": "Young energetic teen, bright tone, fast pace, playful rising intonation, light laughter between phrases, vivid emphasis on keywords.",
                                "Elderly Hoarse": "Elderly male, ~70, slightly hoarse and breathy, slow pace, reflective mood, soft volume, longer pauses, subtle trembling on sustained vowels.",
                                "Audiobook Narrator": "Audiobook narrator, 40s, cinematic and immersive, controlled dynamics, clear phrasing, dramatic pauses, rich low-mid register, smooth resonance.",
                            }
                            def _qwen_get_user_did_from_state(state_params):
                                try:
                                    if isinstance(state_params, dict):
                                        user = state_params.get("user", None)
                                        if user is not None and hasattr(user, "get_did"):
                                            return user.get_did()
                                except Exception:
                                    pass
                                try:
                                    return shared.token.get_guest_did()
                                except Exception:
                                    return None

                            def _qwen_safe_preset_name(name):
                                s = "" if name is None else str(name).strip()
                                s = re.sub(r"[\\/:*?\"<>|\r\n\t]", "_", s)
                                s = s.strip(" .")
                                return s[:80]

                            def _qwen_character_presets_dir(user_did):
                                try:
                                    base = shared.token.get_path_in_user_dir(user_did or shared.token.get_guest_did(), "presets")
                                    path = os.path.join(base, "characters")
                                    os.makedirs(path, exist_ok=True)
                                    return path
                                except Exception:
                                    return None

                            def _qwen_load_user_character_presets(user_did):
                                presets = {}
                                preset_dir = _qwen_character_presets_dir(user_did)
                                if not preset_dir or not os.path.isdir(preset_dir):
                                    return presets
                                try:
                                    for file_name in os.listdir(preset_dir):
                                        if not file_name.lower().endswith(".json"):
                                            continue
                                        full_path = os.path.join(preset_dir, file_name)
                                        if not os.path.isfile(full_path):
                                            continue
                                        try:
                                            with open(full_path, "r", encoding="utf-8") as f:
                                                payload = json.load(f)
                                        except Exception:
                                            continue
                                        key = os.path.splitext(file_name)[0]
                                        text = ""
                                        if isinstance(payload, dict):
                                            key = payload.get("name", key)
                                            text = payload.get("instruction", "")
                                        elif isinstance(payload, str):
                                            text = payload
                                        key = "" if key is None else str(key).strip()
                                        text = "" if text is None else str(text).strip()
                                        if key and text:
                                            presets[key] = text
                                except Exception:
                                    pass
                                return presets

                            def _qwen_get_style_preset_choices(state_params):
                                base_keys = list(qwen_tts_style_presets.keys())
                                user_did = _qwen_get_user_did_from_state(state_params)
                                user_presets = _qwen_load_user_character_presets(user_did)
                                extra = [k for k in sorted(user_presets.keys()) if k not in base_keys]
                                return base_keys + extra

                            def _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value):
                                choices = _qwen_get_style_preset_choices(state_params)
                                dv = None if design_value not in choices else design_value
                                cv = None if custom_value not in choices else custom_value
                                return gr.update(choices=choices, value=dv), gr.update(choices=choices, value=cv)
                            with gr.Row():
                                with gr.Column(scale=4):
                                    qwen_design_instruct = gr.Textbox(label="Style Instruction", lines=4, placeholder="e.g. A cheerful young woman...")
                                with gr.Column(scale=1):
                                    qwen_design_expand_btn = gr.Button(value="Style Expand", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70, visible=MiniCPM.get_enable())
                                    qwen_design_style_preset_choices = gr.Dropdown(label="Character Presets", choices=list(qwen_tts_style_presets.keys()), value=None, show_label=True, elem_classes="qwen_tts_stack_item")
                            with gr.Row():
                                with gr.Column(scale=4):
                                    qwen_design_style_preset_name = gr.Textbox(label="Character Name", lines=1, placeholder="Character Name for Your Role/Style")
                                with gr.Column(scale=1, elem_classes="qwen_tts_preset_stack"):
                                    qwen_design_style_preset_save_btn = gr.Button(value="Save Character", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70)
                                    qwen_design_style_preset_delete_btn = gr.Button(value="Delete Character", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70)
                            with gr.Row():
                                qwen_design_lock_timbre = gr.Checkbox(label="Lock Timbre (clone from first segment)", value=True)
                                qwen_design_clone_batch_size = gr.Slider(label="Batch size", minimum=1, maximum=16, step=1, value=4)
                            with gr.Row():
                                qwen_design_btn = gr.Button("Generate Audio", elem_classes="type_row_half")
                                qwen_design_stop_btn = gr.Button("Stop", elem_classes="type_row_half", min_width=70, visible=False)
                            with gr.Row():
                                with gr.Column(scale=5):
                                    qwen_design_output = gr.Audio(label="Output Audio", interactive=False, show_edit_button=False)
                                with gr.Column(scale=2):
                                    qwen_design_send_target = gr.Dropdown(label="Send To", choices=qwen_send_target_choices, value=None)
                                    qwen_design_send_btn = gr.Button("Send", size="sm")
                            qwen_design_info = gr.Markdown(value="")
                        
                        with gr.Tab("Voice Clone"):
                            qwen_clone_ref_audio = gr.Audio(label="Reference Audio", source="upload", type="numpy")
                            qwen_clone_ref_text = gr.Textbox(label="Reference Audio Text", lines=3, placeholder="Recommended: the spoken content in reference audio")
                            qwen_clone_target_text = gr.Textbox(label="Target Text to Speech", lines=3, placeholder="Enter text here...[pause=800ms] or [pause=0.8s] can add pause between sentences.")
                            qwen_clone_batch_size = gr.Slider(label="Batch size", minimum=1, maximum=16, step=1, value=4)
                            with gr.Row():
                                qwen_clone_btn = gr.Button("Clone & Generate", elem_classes="type_row_half")
                                qwen_clone_stop_btn = gr.Button("Stop", elem_classes="type_row_half", min_width=70, visible=False)
                            with gr.Row():
                                with gr.Column(scale=5):
                                    qwen_clone_output = gr.Audio(label="Output Audio", interactive=False, show_edit_button=False)
                                with gr.Column(scale=2):
                                    qwen_clone_send_target = gr.Dropdown(label="Send To", choices=qwen_send_target_choices, value=None)
                                    qwen_clone_send_btn = gr.Button("Send", size="sm")
                            qwen_clone_info = gr.Markdown(value="")

                        with gr.Tab("Custom Voice"):
                            qwen_custom_text = gr.Textbox(label="Text to Speech", lines=5, placeholder="Enter text here...[pause=800ms] or [pause=0.8s] can add pause between sentences.")
                            _qwen_speaker_notes = {"Serena": ("苏瑶", "中文", "其实我真的有发现，我是一个特别善于观察别人情绪的人。"), "Uncle_fu": ("福伯", "中文", "叶师傅，切他的中路"), "Vivian": ("十三", "中文", "这事情看上去很复杂，其实一点都不简单。"), "Aiden": ("艾登", "英文", "Then by the end of the movie, I got a little bit teary."), "Ryan": ("甜茶", "英文", "Then by the end of the movie, I got a little bit teary."), "Ono_anna": ("小野杏", "日语", "やばい、明日のプレゼン資料まだ完成してない… 助けて！"), "Sohee": ("素熙", "韩语", "야, 오늘 점심에 뭐 먹을지 생각해 봤어? 근처에 새로 생긴 분식집 어때?"), "Dylan": ("晓东", "中文方言-北京话", "我们就在山上啊，就是其实也没什么，就是在土坡上跑来跑去。"), "Eric": ("程川", "中文方言-四川话", "你龟儿太过分了，把我的东西都搞坏了，还晓不晓得认错。")}
                            _qwen_speaker_display_to_key = {"艾登 Aiden": "Aiden", "晓东 Dylan": "Dylan", "程川 Eric": "Eric", "小野杏 Ono Anna": "Ono_anna", "甜茶 Ryan": "Ryan", "苏瑶 Serena": "Serena", "素熙 Sohee": "Sohee", "福伯 Uncle Fu": "Uncle_fu", "十三 Vivian": "Vivian"}
                            _qwen_default_speaker_display = "甜茶 Ryan"

                            def _qwen_speaker_key(speaker_value):
                                v = speaker_value[1] if isinstance(speaker_value, (list, tuple)) and len(speaker_value) >= 2 else speaker_value
                                s = "" if v is None else str(v).strip()
                                s = _qwen_speaker_display_to_key.get(s, s)
                                if s.startswith("(") and s.endswith(")"):
                                    try:
                                        import ast
                                        p = ast.literal_eval(s)
                                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                                            s = "" if p[1] is None else str(p[1]).strip()
                                    except Exception:
                                        pass
                                s = _qwen_speaker_display_to_key.get(s, s)
                                if s not in _qwen_speaker_notes and " " in s:
                                    c = s.split()[-1].strip()
                                    if c in _qwen_speaker_notes:
                                        s = c
                                return s

                            def _format_qwen_speaker_note(speaker_value: str):
                                speaker_key = _qwen_speaker_key(speaker_value)
                                note = _qwen_speaker_notes.get(speaker_key, None)
                                if not note:
                                    return "在左侧选择说话人后，这里会显示角色介绍。"
                                alias, lang, text = note
                                title = f"{alias} {speaker_key}".strip()
                                return f"**音色**：{title}\n\n**语种**：{lang}\n\n**合成文本示例**：{text}"

                            with gr.Row():
                                with gr.Column(scale=3):
                                    qwen_custom_speaker = gr.Dropdown(label="Speaker", choices=list(_qwen_speaker_display_to_key.keys()), value=_qwen_default_speaker_display)
                                with gr.Column(scale=3):
                                    qwen_custom_speaker_note = gr.Markdown(value=_format_qwen_speaker_note(_qwen_speaker_display_to_key[_qwen_default_speaker_display]))
                            with gr.Row():
                                with gr.Column(scale=4):
                                    qwen_custom_instruct = gr.Textbox(label="Style Instruction (Optional)", lines=4)
                                with gr.Column(scale=1):
                                    qwen_custom_expand_btn = gr.Button(value="Style Expand", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70, visible=MiniCPM.get_enable())
                                    qwen_custom_style_preset_choices = gr.Dropdown(label="Character Presets", choices=list(qwen_tts_style_presets.keys()), value=None, show_label=True, elem_classes="qwen_tts_stack_item")
                            qwen_custom_batch_size = gr.Slider(label="Batch size", minimum=1, maximum=16, step=1, value=4)
                            with gr.Row():
                                with gr.Column(scale=4):
                                    qwen_custom_style_preset_name = gr.Textbox(label="Character Name", lines=1, placeholder="Character Name for Your Role/Style", elem_classes="qwen_tts_stack_item")
                                with gr.Column(scale=1, elem_classes="qwen_tts_preset_stack"):   
                                    qwen_custom_style_preset_save_btn = gr.Button(value="Save Character", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70)
                                    qwen_custom_style_preset_delete_btn = gr.Button(value="Delete Character", elem_classes=["type_row_half", "qwen_tts_stack_item"], size="sm", min_width=70)
                            with gr.Row():
                                qwen_custom_btn = gr.Button("Generate Audio", elem_classes="type_row_half")
                                qwen_custom_stop_btn = gr.Button("Stop", elem_classes="type_row_half", min_width=70, visible=False)
                            with gr.Row():
                                with gr.Column(scale=5):
                                    qwen_custom_output = gr.Audio(label="Output Audio", interactive=False, show_edit_button=False)
                                with gr.Column(scale=2):
                                    qwen_custom_send_target = gr.Dropdown(label="Send To", choices=qwen_send_target_choices, value=None)
                                    qwen_custom_send_btn = gr.Button("Send", size="sm")
                            qwen_custom_info = gr.Markdown(value="")

                        with gr.Tab("Dialogue"):
                            qwen_dialogue_script = gr.Textbox(label="Script", lines=8, placeholder="Format: 角色名: 文本（每行一句）\n\n角色1: 你好，今天我们聊点什么？\n角色2: 我想了解一下 Qwen3-TTS 的语音克隆。\n角色3: 我来总结参数设置要点。\n旁白: 他们开始了一段轻松的对话。")
                            def _qwen_dialogue_role(role_label, default_name):
                                with gr.Column():
                                    name = gr.Textbox(label=f"{role_label} Name", value=default_name)
                                    audio = gr.Audio(label=f"{role_label} Reference Audio", source="upload", type="numpy")
                                    ref_text = gr.Textbox(label=f"{role_label} Reference Text", lines=2)
                                return name, audio, ref_text
                            with gr.Row():
                                qwen_role_1_name, qwen_role_1_audio, qwen_role_1_ref_text = _qwen_dialogue_role("Role 1", "角色1")
                                qwen_role_2_name, qwen_role_2_audio, qwen_role_2_ref_text = _qwen_dialogue_role("Role 2", "角色2")
                            with gr.Row():
                                qwen_role_3_name, qwen_role_3_audio, qwen_role_3_ref_text = _qwen_dialogue_role("Role 3", "角色3")
                                qwen_role_4_name, qwen_role_4_audio, qwen_role_4_ref_text = _qwen_dialogue_role("Role 4", "旁白")
                            with gr.Row():
                                qwen_dialogue_btn = gr.Button("Generate Dialogue Audio", elem_classes="type_row_half")
                                qwen_dialogue_stop_btn = gr.Button("Stop", elem_classes="type_row_half", min_width=70, visible=False)
                            with gr.Row():
                                with gr.Column(scale=5):
                                    qwen_dialogue_output = gr.Audio(label="Output Audio", interactive=False, show_edit_button=False)
                                with gr.Column(scale=2):
                                    qwen_dialogue_send_target = gr.Dropdown(label="Send To", choices=qwen_send_target_choices, value=None)
                                    qwen_dialogue_send_btn = gr.Button("Send", size="sm")
                            qwen_dialogue_info = gr.Markdown(value="")
                        
                        with gr.Tab("Settings"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_model_size = gr.Radio(["0.6B", "1.7B"], label="Model Size", value="1.7B")
                                with gr.Column(scale=1):
                                    qwen_tts_precision = gr.Radio(["bf16", "fp32"], label="Precision", value="bf16")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_device = gr.Dropdown(label="Device", choices=["auto", "cuda", "mps", "cpu"], value="auto")
                                with gr.Column(scale=1):
                                    qwen_tts_language = gr.Dropdown(label="Language", choices=["Auto", "Chinese", "English", "Japanese", "Korean"], value="Auto")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_attention = gr.Dropdown(label="Attention", choices=["auto", "sage_attn", "flash_attn", "sdpa", "eager"], value="auto")
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        qwen_tts_seed_random = gr.Checkbox(label="Random", value=True)
                                        qwen_tts_seed = gr.Number(label="Seed", value=0, precision=0)
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_max_new_tokens = gr.Slider(label="Max new tokens", minimum=512, maximum=16384, step=256, value=4096)
                                with gr.Column(scale=1):
                                    qwen_tts_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=1.0)
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_split_max_chars = gr.Slider(label="Split max chars", minimum=20, maximum=600, step=10, value=200)
                                with gr.Column(scale=1):
                                    qwen_tts_split_hard_max_chars = gr.Slider(label="Split hard max chars", minimum=20, maximum=800, step=10, value=260)
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.05, value=0.8)
                                with gr.Column(scale=1):
                                    qwen_tts_top_k = gr.Slider(label="Top-k", minimum=0, maximum=100, step=1, value=20)
                            with gr.Row():
                                with gr.Column(scale=1):
                                    qwen_tts_repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.05)
                                with gr.Column(scale=1):
                                    qwen_tts_unload = gr.Checkbox(label="Unload model after generate", value=True)
                            with gr.Accordion("Dialogue Pause Settings", open=False):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        qwen_pause_linebreak = gr.Slider(label="Linebreak pause", minimum=0.0, maximum=5.0, step=0.1, value=0.5)
                                    with gr.Column(scale=1):
                                        qwen_period_pause = gr.Slider(label="Period pause (.)", minimum=0.0, maximum=5.0, step=0.1, value=0.4)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        qwen_comma_pause = gr.Slider(label="Comma pause (,)", minimum=0.0, maximum=5.0, step=0.1, value=0.2)
                                    with gr.Column(scale=1):
                                        qwen_question_pause = gr.Slider(label="Question pause (?)", minimum=0.0, maximum=5.0, step=0.1, value=0.6)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        qwen_hyphen_pause = gr.Slider(label="Hyphen pause (-)", minimum=0.0, maximum=5.0, step=0.1, value=0.3)
                                    with gr.Column(scale=1):
                                        qwen_dialogue_merge = gr.Checkbox(label="Merge outputs", value=True)
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        qwen_dialogue_batch = gr.Slider(label="Batch size", minimum=1, maximum=32, step=1, value=4)
                                    with gr.Column(scale=1):
                                        qwen_dialogue_max_tokens = gr.Slider(label="Max new tokens per line", minimum=512, maximum=8192, step=256, value=4096)
                    gr.HTML(
                        value='项目来源：<a href="https://www.modelscope.cn/collections/Qwen/Qwen3-TTS" target="_blank" rel="noopener noreferrer">https://www.modelscope.cn/collections/Qwen/Qwen3-TTS</a>',
                        elem_id="qwen_tts_source_badge",
                    )

                    try:
                        from enhanced import webui_qwen_tts

                        def _get_user_did_from_state(state_params): return _qwen_get_user_did_from_state(state_params)

                        def _resolve_tts_seed(seed_value, seed_random_value):
                            try:
                                seed_int = int(seed_value)
                            except Exception:
                                seed_int = 0
                            if seed_random_value:
                                return random.randint(0, 2147483647)
                            return seed_int

                        def _is_blank(value):
                            if value is None:
                                return True
                            try:
                                return str(value).strip() == ""
                            except Exception:
                                return True

                        def _qwen_tts_set_interrupt(value: bool):
                            try:
                                model_management.interrupt_current_processing(bool(value))
                            except Exception:
                                pass
                            try:
                                from comfy import model_management as comfy_model_management
                                comfy_model_management.interrupt_current_processing(bool(value))
                            except Exception:
                                pass

                        qwen_tts_force_unload = {"flag": False}

                        def _qwen_tts_begin():
                            _qwen_tts_set_interrupt(False)
                            qwen_tts_force_unload["flag"] = False
                            return gr.update(visible=False), gr.update(visible=True), "生成中…"

                        def _qwen_tts_end():
                            _qwen_tts_set_interrupt(False)
                            return gr.update(visible=True), gr.update(visible=False)

                        def _qwen_tts_stop():
                            _qwen_tts_set_interrupt(True)
                            qwen_tts_force_unload["flag"] = True
                            return "正在停止…"

                        def _qwen_is_interrupt_exception(e: Exception) -> bool:
                            if type(e).__name__ == "InterruptProcessingException":
                                return True
                            try:
                                if isinstance(e, model_management.InterruptProcessingException):
                                    return True
                            except Exception:
                                pass
                            try:
                                from comfy import model_management as comfy_model_management
                                if isinstance(e, comfy_model_management.InterruptProcessingException):
                                    return True
                            except Exception:
                                pass
                            return False

                        qwen_tts_seed_random.change(fn=lambda is_random: gr.update(interactive=not bool(is_random)), inputs=[qwen_tts_seed_random], outputs=[qwen_tts_seed], queue=False, show_progress=False)

                        qwen_custom_speaker.change(fn=_format_qwen_speaker_note, inputs=[qwen_custom_speaker], outputs=[qwen_custom_speaker_note], queue=False, show_progress=False)

                        qwen_send_target_key_to_index = {
                            "qwen_clone_ref_audio": 0,
                            "qwen_role_1_audio": 1,
                            "qwen_role_2_audio": 2,
                            "qwen_role_3_audio": 3,
                            "qwen_role_4_audio": 4,
                            "scene_audio": 5,
                        }
                        qwen_send_numpy_target_keys = {
                            "qwen_clone_ref_audio",
                            "qwen_role_1_audio",
                            "qwen_role_2_audio",
                            "qwen_role_3_audio",
                            "qwen_role_4_audio",
                        }

                        def _qwen_read_wav_file(audio_path: str):
                            p = "" if audio_path is None else str(audio_path).strip()
                            if not p:
                                return None
                            if not os.path.isfile(p):
                                return None
                            try:
                                with wave.open(p, "rb") as wf:
                                    sr = int(wf.getframerate())
                                    channels = int(wf.getnchannels())
                                    sample_width = int(wf.getsampwidth())
                                    frames = wf.readframes(int(wf.getnframes()))
                                if sample_width == 2:
                                    data = np.frombuffer(frames, dtype=np.int16)
                                elif sample_width == 4:
                                    data32 = np.frombuffer(frames, dtype=np.int32)
                                    data = (data32 / 65536.0).astype(np.int16)
                                else:
                                    return None
                                if channels > 1:
                                    data = data.reshape(-1, channels)
                                return sr, data
                            except Exception:
                                return None

                        def _qwen_write_wav_temp(sr: int, wav):
                            try:
                                sample_rate = int(sr)
                            except Exception:
                                return None
                            try:
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
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                    out_path = os.path.abspath(tmp.name)
                                with wave.open(out_path, "wb") as wf:
                                    wf.setnchannels(channels)
                                    wf.setsampwidth(2)
                                    wf.setframerate(sample_rate)
                                    wf.writeframes(audio.tobytes())
                                return out_path
                            except Exception:
                                return None

                        def _qwen_audio_to_numpy(audio):
                            if audio is None:
                                return None
                            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                                sr = audio.get("sample_rate", None)
                                wav = audio.get("waveform", None)
                                if sr is None or wav is None:
                                    return None
                                if hasattr(wav, "cpu"):
                                    wav = wav.cpu().numpy()
                                wav = np.asarray(wav)
                                wav = np.squeeze(wav)
                                if wav.dtype == np.float32:
                                    wav = np.clip(wav, -1.0, 1.0)
                                    wav = (wav * 32767.0).astype(np.int16)
                                return int(sr), wav
                            if isinstance(audio, (tuple, list)) and len(audio) == 2:
                                sr, wav = audio
                                if sr is None or wav is None:
                                    return None
                                if hasattr(wav, "cpu"):
                                    wav = wav.cpu().numpy()
                                return int(sr), np.asarray(wav)
                            if isinstance(audio, str):
                                return _qwen_read_wav_file(audio)
                            return None

                        def _qwen_audio_to_filepath(audio):
                            if audio is None:
                                return None
                            if isinstance(audio, str):
                                p = "" if audio is None else str(audio).strip()
                                return p if p else None
                            as_numpy = _qwen_audio_to_numpy(audio)
                            if as_numpy is None:
                                return None
                            sr, wav = as_numpy
                            return _qwen_write_wav_temp(sr, wav)

                        def _qwen_send_audio_to_target(output_audio, target_label):
                            outputs = [gr.update()] * 6
                            if _is_blank(target_label):
                                gr.Warning("请选择要覆盖的 Audio 控件。")
                                return outputs
                            target_key = qwen_send_target_options.get(str(target_label).strip(), None)
                            if not target_key:
                                gr.Warning("未识别的目标 Audio 控件。")
                                return outputs
                            if output_audio is None:
                                gr.Warning("当前没有可发送的输出音频。")
                                return outputs
                            if target_key in qwen_send_numpy_target_keys:
                                value = _qwen_audio_to_numpy(output_audio)
                            else:
                                value = _qwen_audio_to_filepath(output_audio)
                            if value is None:
                                gr.Warning("音频格式转换失败，无法覆盖目标。")
                                return outputs
                            idx = qwen_send_target_key_to_index.get(target_key, None)
                            if idx is None:
                                gr.Warning("目标 Audio 控件索引异常。")
                                return outputs
                            outputs[idx] = value
                            return outputs

                        def _apply_style_presets(selected_preset, state_params):
                            if _is_blank(selected_preset):
                                return ""
                            k = str(selected_preset).strip()
                            v = qwen_tts_style_presets.get(k, None)
                            if v is not None:
                                return str(v).strip()
                            user_did = _get_user_did_from_state(state_params)
                            user_presets = _qwen_load_user_character_presets(user_did)
                            return str(user_presets.get(k, "")).strip()

                        qwen_design_style_preset_choices.change(fn=_apply_style_presets, inputs=[qwen_design_style_preset_choices, state_topbar], outputs=[qwen_design_instruct], queue=False, show_progress=False)
                        qwen_custom_style_preset_choices.change(fn=_apply_style_presets, inputs=[qwen_custom_style_preset_choices, state_topbar], outputs=[qwen_custom_instruct], queue=False, show_progress=False)

                        def _save_user_character_preset(preset_name, style_text, state_params, design_value, custom_value, target):
                            user_did = _get_user_did_from_state(state_params)
                            name = _qwen_safe_preset_name(preset_name)
                            text = "" if style_text is None else str(style_text).strip()
                            if not name:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "Preset Name 不能为空")
                            if not text:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "Style Instruction 不能为空")
                            preset_dir = _qwen_character_presets_dir(user_did)
                            if not preset_dir:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "保存失败：无法定位用户目录")
                            file_path = os.path.join(preset_dir, f"{name}.json")
                            payload = {"name": name, "instruction": text, "updated_at": int(time.time())}
                            try:
                                with open(file_path, "w", encoding="utf-8") as f:
                                    json.dump(payload, f, ensure_ascii=False, indent=2)
                            except Exception as e:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), f"保存失败：{type(e).__name__}: {e}")
                            dv = name if str(target) == "design" else design_value
                            cv = name if str(target) == "custom" else custom_value
                            return _qwen_refresh_style_preset_dropdowns(state_params, dv, cv) + (gr.update(value=""), f"已保存到 users/{user_did}/presets/characters/{name}.json")

                        def _delete_user_character_preset(preset_name, state_params, design_value, custom_value, target):
                            user_did = _get_user_did_from_state(state_params)
                            selected = design_value if str(target) == "design" else custom_value
                            name = _qwen_safe_preset_name(preset_name)
                            if not name:
                                name = _qwen_safe_preset_name(selected)
                            if not name:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "Preset Name 不能为空")
                            if name in qwen_tts_style_presets:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "不能删除内置 Preset")
                            preset_dir = _qwen_character_presets_dir(user_did)
                            if not preset_dir:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), "删除失败：无法定位用户目录")
                            file_path = os.path.join(preset_dir, f"{name}.json")
                            if not os.path.isfile(file_path):
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), f"未找到 Preset：{name}")
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                return _qwen_refresh_style_preset_dropdowns(state_params, design_value, custom_value) + (gr.update(value=preset_name), f"删除失败：{type(e).__name__}: {e}")
                            dv = None if str(target) == "design" and str(design_value).strip() == name else design_value
                            cv = None if str(target) == "custom" and str(custom_value).strip() == name else custom_value
                            return _qwen_refresh_style_preset_dropdowns(state_params, dv, cv) + (gr.update(value=""), f"已删除 users/{user_did}/presets/characters/{name}.json")

                        qwen_design_style_preset_save_btn.click(fn=lambda a, b, c, d, e: _save_user_character_preset(a, b, c, d, e, "design"), inputs=[qwen_design_style_preset_name, qwen_design_instruct, state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices, qwen_design_style_preset_name, qwen_design_info], queue=False, show_progress=False)
                        qwen_custom_style_preset_save_btn.click(fn=lambda a, b, c, d, e: _save_user_character_preset(a, b, c, d, e, "custom"), inputs=[qwen_custom_style_preset_name, qwen_custom_instruct, state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices, qwen_custom_style_preset_name, qwen_custom_info], queue=False, show_progress=False)
                        qwen_design_style_preset_delete_btn.click(fn=lambda a, b, c, d: _delete_user_character_preset(a, b, c, d, "design"), inputs=[qwen_design_style_preset_name, state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices, qwen_design_style_preset_name, qwen_design_info], queue=False, show_progress=False)
                        qwen_custom_style_preset_delete_btn.click(fn=lambda a, b, c, d: _delete_user_character_preset(a, b, c, d, "custom"), inputs=[qwen_custom_style_preset_name, state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices, qwen_custom_style_preset_name, qwen_custom_info], queue=False, show_progress=False)

                        def _expand_tts_style_instruction(style_text, state_params):
                            if _is_blank(style_text):
                                return style_text, "请先输入 Style Instruction，再进行风格扩展。"
                            if not MiniCPM.get_enable():
                                return style_text, "请先在 Identity -> Local System 启用 VLM。"
                            if not minicpm.model_exists():
                                return style_text, "VLM 模型未就绪，请先下载/配置 VLM 模型。"
                            try:
                                with worker.external_exclusive_task():
                                    expanded = minicpm.expand_tts_style_instruction(style_text)
                                expanded = str(expanded).strip() if expanded is not None else ""
                                if not expanded:
                                    return style_text, "风格扩展未返回有效内容。"
                                return expanded, ""
                            except Exception as e:
                                return style_text, f"风格扩展失败：{type(e).__name__}: {e}"

                        qwen_design_expand_btn.click(fn=_expand_tts_style_instruction, inputs=[qwen_design_instruct, state_topbar], outputs=[qwen_design_instruct, qwen_design_info], queue=False, show_progress=True)
                        qwen_custom_expand_btn.click(fn=_expand_tts_style_instruction, inputs=[qwen_custom_instruct, state_topbar], outputs=[qwen_custom_instruct, qwen_custom_info], queue=False, show_progress=True)

                        def _qwen_after_unload(unload):
                            if not bool(unload):
                                return
                            try:
                                webui_qwen_tts.unload_qwen_tts_models()
                            except Exception:
                                pass
                            try:
                                unload_models_clicked(False)
                            except Exception:
                                pass

                        def _qwen_tts_cleanup(unload):
                            should_unload = bool(unload) or bool(qwen_tts_force_unload.get("flag"))
                            qwen_tts_force_unload["flag"] = False
                            if not should_unload:
                                return
                            _qwen_after_unload(True)

                        def _qwen_call_progress(handler_fn, seed_random, seed, unload, state_params, **kwargs):
                            import queue as _queue
                            import threading as _threading
                            import time as _time

                            try:
                                seed_int = int(seed)
                            except Exception:
                                seed_int = 0

                            used_seed = None
                            try:
                                used_seed = _resolve_tts_seed(seed, seed_random)
                            except Exception:
                                used_seed = seed_int

                            q: _queue.Queue = _queue.Queue()
                            done = {"flag": False}
                            out = {"audio_path": None, "error": None}

                            def progress_callback(pct, msg=""):
                                try:
                                    p = int(pct)
                                except Exception:
                                    p = 0
                                if p < 0:
                                    p = 0
                                if p > 100:
                                    p = 100
                                try:
                                    q.put((p, str(msg or "").strip()), block=False)
                                except Exception:
                                    pass

                            def runner():
                                try:
                                    audio_path = webui_qwen_tts.enqueue_task(
                                        handler_fn,
                                        user_did=_get_user_did_from_state(state_params),
                                        seed=int(used_seed),
                                        progress_callback=progress_callback,
                                        **kwargs,
                                    )
                                    out["audio_path"] = audio_path
                                except Exception as e:
                                    out["error"] = e
                                finally:
                                    done["flag"] = True
                                    try:
                                        q.put(None, block=False)
                                    except Exception:
                                        pass

                            _threading.Thread(target=runner, daemon=True).start()

                            last_pct = 0.0
                            last_msg = "生成中…"
                            last_tick = _time.time()
                            last_pct_is_synthetic = False

                            def _fmt_pct(p):
                                try:
                                    pv = float(p)
                                except Exception:
                                    pv = 0.0
                                if pv >= 95.0:
                                    return f"{pv:.1f}%"
                                return f"{int(pv)}%"

                            yield gr.update(), used_seed, f"{last_msg} {_fmt_pct(last_pct)}"

                            while True:
                                if done["flag"]:
                                    break
                                try:
                                    item = q.get(timeout=0.25)
                                except _queue.Empty:
                                    item = None
                                if item is None:
                                    now = _time.time()
                                    if last_pct < 95.0:
                                        interval_s = 3.0
                                        step = 1.0
                                        cap = 95.0
                                    else:
                                        interval_s = 1.0
                                        step = 0.1
                                        cap = 99.9
                                    if now - last_tick >= interval_s and last_pct < cap:
                                        last_tick = now
                                        last_pct = min(cap, last_pct + step)
                                        last_pct_is_synthetic = True
                                        yield gr.update(), used_seed, f"{last_msg} {_fmt_pct(last_pct)}"
                                    continue
                                pct, msg = item
                                if msg:
                                    last_msg = msg
                                if last_pct_is_synthetic:
                                    last_pct = float(pct)
                                else:
                                    if pct < last_pct:
                                        pct = last_pct
                                    last_pct = float(pct)
                                last_pct_is_synthetic = False
                                last_tick = _time.time()
                                yield gr.update(), used_seed, f"{last_msg} {_fmt_pct(last_pct)}"
                                _time.sleep(0.01)

                            err = out.get("error")
                            if err is None:
                                yield out.get("audio_path"), used_seed, gr.update()
                                return

                            interrupted = _qwen_is_interrupt_exception(err)
                            if interrupted:
                                yield gr.update(value=None), seed_int, "已中断。"
                                return
                            yield gr.update(value=None), seed_int, f"生成失败：{type(err).__name__}: {err}"
                            return

                        def qwen_voice_design_fn(text, instruct, model_choice, precision, device, language, seed_random, seed, max_new_tokens, split_max_chars, split_hard_max_chars, top_p, top_k, temperature, repetition_penalty, attention, unload, lock_timbre, clone_batch_size, state_params):
                            try:
                                seed_int = int(seed)
                            except Exception:
                                seed_int = 0
                            if _is_blank(text):
                                yield gr.update(value=None), seed_int, "请输入“Text to Speech”后再生成。"
                                return
                            yield from _qwen_call_progress(webui_qwen_tts.qwen_tts_handler.voice_design, seed_random, seed, unload, state_params, text=text, instruct=instruct, model_choice=model_choice, device=device, precision=precision, language=language, max_new_tokens=int(max_new_tokens), max_chars=int(split_max_chars), hard_max_chars=int(split_hard_max_chars), top_p=float(top_p), top_k=int(top_k), temperature=float(temperature), repetition_penalty=float(repetition_penalty), attention=attention, unload_model_after_generate=bool(unload), lock_timbre_with_first_segment=bool(lock_timbre), clone_batch_size=int(clone_batch_size))

                        qwen_design_btn.click(fn=_qwen_tts_begin, inputs=[], outputs=[qwen_design_btn, qwen_design_stop_btn, qwen_design_info], queue=False, show_progress=False).then(fn=qwen_voice_design_fn, inputs=[qwen_design_text, qwen_design_instruct, qwen_tts_model_size, qwen_tts_precision, qwen_tts_device, qwen_tts_language, qwen_tts_seed_random, qwen_tts_seed, qwen_tts_max_new_tokens, qwen_tts_split_max_chars, qwen_tts_split_hard_max_chars, qwen_tts_top_p, qwen_tts_top_k, qwen_tts_temperature, qwen_tts_repetition_penalty, qwen_tts_attention, qwen_tts_unload, qwen_design_lock_timbre, qwen_design_clone_batch_size, state_topbar], outputs=[qwen_design_output, qwen_tts_seed, qwen_design_info], queue=True, show_progress=False).then(fn=_qwen_tts_end, inputs=[], outputs=[qwen_design_btn, qwen_design_stop_btn], queue=False, show_progress=False).then(fn=_qwen_tts_cleanup, inputs=[qwen_tts_unload], queue=False, show_progress=False)
                        qwen_design_stop_btn.click(fn=_qwen_tts_stop, inputs=[], outputs=[qwen_design_info], queue=False, show_progress=False)

                        def qwen_voice_clone_fn(ref_audio, ref_text, target_text, model_choice, precision, device, language, seed_random, seed, max_new_tokens, split_max_chars, split_hard_max_chars, top_p, top_k, temperature, repetition_penalty, attention, unload, batch_size, state_params):
                            try:
                                seed_int = int(seed)
                            except Exception:
                                seed_int = 0
                            if ref_audio is None:
                                yield gr.update(value=None), seed_int, "请先上传“Reference Audio”。"
                                return
                            if _is_blank(target_text):
                                yield gr.update(value=None), seed_int, "请输入“Target Text to Speech”后再生成。"
                                return
                            yield from _qwen_call_progress(webui_qwen_tts.qwen_tts_handler.voice_clone, seed_random, seed, unload, state_params, ref_audio=ref_audio, ref_text=ref_text, target_text=target_text, model_choice=model_choice, device=device, precision=precision, language=language, max_new_tokens=int(max_new_tokens), max_chars=int(split_max_chars), hard_max_chars=int(split_hard_max_chars), top_p=float(top_p), top_k=int(top_k), temperature=float(temperature), repetition_penalty=float(repetition_penalty), x_vector_only=False, attention=attention, unload_model_after_generate=bool(unload), batch_size=int(batch_size))

                        qwen_clone_btn.click(fn=_qwen_tts_begin, inputs=[], outputs=[qwen_clone_btn, qwen_clone_stop_btn, qwen_clone_info], queue=False, show_progress=False).then(fn=qwen_voice_clone_fn, inputs=[qwen_clone_ref_audio, qwen_clone_ref_text, qwen_clone_target_text, qwen_tts_model_size, qwen_tts_precision, qwen_tts_device, qwen_tts_language, qwen_tts_seed_random, qwen_tts_seed, qwen_tts_max_new_tokens, qwen_tts_split_max_chars, qwen_tts_split_hard_max_chars, qwen_tts_top_p, qwen_tts_top_k, qwen_tts_temperature, qwen_tts_repetition_penalty, qwen_tts_attention, qwen_tts_unload, qwen_clone_batch_size, state_topbar], outputs=[qwen_clone_output, qwen_tts_seed, qwen_clone_info], queue=True, show_progress=False).then(fn=_qwen_tts_end, inputs=[], outputs=[qwen_clone_btn, qwen_clone_stop_btn], queue=False, show_progress=False).then(fn=_qwen_tts_cleanup, inputs=[qwen_tts_unload], queue=False, show_progress=False)
                        qwen_clone_stop_btn.click(fn=_qwen_tts_stop, inputs=[], outputs=[qwen_clone_info], queue=False, show_progress=False)

                        def qwen_custom_voice_fn(text, speaker, instruct, model_choice, precision, device, language, seed_random, seed, max_new_tokens, split_max_chars, split_hard_max_chars, top_p, top_k, temperature, repetition_penalty, attention, unload, batch_size, state_params):
                            try:
                                seed_int = int(seed)
                            except Exception:
                                seed_int = 0
                            if _is_blank(text):
                                yield gr.update(value=None), seed_int, "请输入“Text to Speech”后再生成。"
                                return
                            speaker_key = "" if speaker is None else str(speaker).strip()
                            if _is_blank(speaker_key):
                                yield gr.update(value=None), seed_int, "请选择“Speaker”后再生成。"
                                return
                            if speaker_key in _qwen_speaker_display_to_key:
                                speaker_key = _qwen_speaker_display_to_key[speaker_key]
                            yield from _qwen_call_progress(webui_qwen_tts.qwen_tts_handler.custom_voice, seed_random, seed, unload, state_params, text=text, speaker=speaker_key, instruct=instruct, model_choice=model_choice, device=device, precision=precision, language=language, max_new_tokens=int(max_new_tokens), max_chars=int(split_max_chars), hard_max_chars=int(split_hard_max_chars), top_p=float(top_p), top_k=int(top_k), temperature=float(temperature), repetition_penalty=float(repetition_penalty), attention=attention, unload_model_after_generate=bool(unload), custom_model_path="", custom_speaker_name="", batch_size=int(batch_size))

                        qwen_custom_btn.click(fn=_qwen_tts_begin, inputs=[], outputs=[qwen_custom_btn, qwen_custom_stop_btn, qwen_custom_info], queue=False, show_progress=False).then(fn=qwen_custom_voice_fn, inputs=[qwen_custom_text, qwen_custom_speaker, qwen_custom_instruct, qwen_tts_model_size, qwen_tts_precision, qwen_tts_device, qwen_tts_language, qwen_tts_seed_random, qwen_tts_seed, qwen_tts_max_new_tokens, qwen_tts_split_max_chars, qwen_tts_split_hard_max_chars, qwen_tts_top_p, qwen_tts_top_k, qwen_tts_temperature, qwen_tts_repetition_penalty, qwen_tts_attention, qwen_tts_unload, qwen_custom_batch_size, state_topbar], outputs=[qwen_custom_output, qwen_tts_seed, qwen_custom_info], queue=True, show_progress=False).then(fn=_qwen_tts_end, inputs=[], outputs=[qwen_custom_btn, qwen_custom_stop_btn], queue=False, show_progress=False).then(fn=_qwen_tts_cleanup, inputs=[qwen_tts_unload], queue=False, show_progress=False)
                        qwen_custom_stop_btn.click(fn=_qwen_tts_stop, inputs=[], outputs=[qwen_custom_info], queue=False, show_progress=False)

                        def qwen_dialogue_fn(script, r1n, r1a, r1t, r2n, r2a, r2t, r3n, r3a, r3t, r4n, r4a, r4t, model_choice, precision, device, language, seed_random, seed, top_p, top_k, temperature, repetition_penalty, attention, unload, pause_linebreak, period_pause, comma_pause, question_pause, hyphen_pause, merge_outputs, batch_size, max_tokens_per_line, state_params):
                            try:
                                seed_int = int(seed)
                            except Exception:
                                seed_int = 0
                            if _is_blank(script):
                                yield gr.update(value=None), seed_int, "请先填写“Script”（可参考占位示例）。"
                                return
                            yield from _qwen_call_progress(webui_qwen_tts.qwen_tts_handler.dialogue, seed_random, seed, unload, state_params, script=script, role_1_name=r1n, role_1_audio=r1a, role_1_ref_text=r1t, role_2_name=r2n, role_2_audio=r2a, role_2_ref_text=r2t, role_3_name=r3n, role_3_audio=r3a, role_3_ref_text=r3t, role_4_name=r4n, role_4_audio=r4a, role_4_ref_text=r4t, model_choice=model_choice, device=device, precision=precision, language=language, pause_linebreak=float(pause_linebreak), period_pause=float(period_pause), comma_pause=float(comma_pause), question_pause=float(question_pause), hyphen_pause=float(hyphen_pause), merge_outputs=bool(merge_outputs), batch_size=int(batch_size), max_new_tokens_per_line=int(max_tokens_per_line), top_p=float(top_p), top_k=int(top_k), temperature=float(temperature), repetition_penalty=float(repetition_penalty), attention=attention, unload_model_after_generate=bool(unload))

                        qwen_dialogue_btn.click(fn=_qwen_tts_begin, inputs=[], outputs=[qwen_dialogue_btn, qwen_dialogue_stop_btn, qwen_dialogue_info], queue=False, show_progress=False).then(fn=qwen_dialogue_fn, inputs=[qwen_dialogue_script, qwen_role_1_name, qwen_role_1_audio, qwen_role_1_ref_text, qwen_role_2_name, qwen_role_2_audio, qwen_role_2_ref_text, qwen_role_3_name, qwen_role_3_audio, qwen_role_3_ref_text, qwen_role_4_name, qwen_role_4_audio, qwen_role_4_ref_text, qwen_tts_model_size, qwen_tts_precision, qwen_tts_device, qwen_tts_language, qwen_tts_seed_random, qwen_tts_seed, qwen_tts_top_p, qwen_tts_top_k, qwen_tts_temperature, qwen_tts_repetition_penalty, qwen_tts_attention, qwen_tts_unload, qwen_pause_linebreak, qwen_period_pause, qwen_comma_pause, qwen_question_pause, qwen_hyphen_pause, qwen_dialogue_merge, qwen_dialogue_batch, qwen_dialogue_max_tokens, state_topbar], outputs=[qwen_dialogue_output, qwen_tts_seed, qwen_dialogue_info], queue=True, show_progress=False).then(fn=_qwen_tts_end, inputs=[], outputs=[qwen_dialogue_btn, qwen_dialogue_stop_btn], queue=False, show_progress=False).then(fn=_qwen_tts_cleanup, inputs=[qwen_tts_unload], queue=False, show_progress=False)
                        qwen_dialogue_stop_btn.click(fn=_qwen_tts_stop, inputs=[], outputs=[qwen_dialogue_info], queue=False, show_progress=False)

                        qwen_send_outputs = [qwen_clone_ref_audio, qwen_role_1_audio, qwen_role_2_audio, qwen_role_3_audio, qwen_role_4_audio, scene_audio]
                        qwen_design_send_btn.click(fn=_qwen_send_audio_to_target, inputs=[qwen_design_output, qwen_design_send_target], outputs=qwen_send_outputs, queue=False, show_progress=False)
                        qwen_clone_send_btn.click(fn=_qwen_send_audio_to_target, inputs=[qwen_clone_output, qwen_clone_send_target], outputs=qwen_send_outputs, queue=False, show_progress=False)
                        qwen_custom_send_btn.click(fn=_qwen_send_audio_to_target, inputs=[qwen_custom_output, qwen_custom_send_target], outputs=qwen_send_outputs, queue=False, show_progress=False)
                        qwen_dialogue_send_btn.click(fn=_qwen_send_audio_to_target, inputs=[qwen_dialogue_output, qwen_dialogue_send_target], outputs=qwen_send_outputs, queue=False, show_progress=False)

                    except ImportError:
                        print("Warning: webui_qwen_tts module not found. TTS features disabled.")
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id, elem_id='image_input_tabs'):
                    with gr.Tab(label='Image Prompt', id='ip_tab', elem_id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_advanced = gr.Checkbox(label='Advanced Control', value=modules.config.default_image_prompt_advanced_checkbox, container=False, scale=5, visible=False)
                            ip_auto_detect = gr.Checkbox(label='Auto Detect Control Image Type', value=True, container=False, scale=5, elem_id='ip_auto_detect')
                            preview_preprocessing = gr.Button(value='💥Preview Preprocessor', scale=1)
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            ip_detect_style_nodes = []
                            ip_image_elem_ids = []

                            def _ip_make_auto_detect_fn(elem_id: str):
                                def _fn(image_np, selected_type, enabled):
                                    if image_np is None:
                                        return '', gr.update(), gr.update(), gr.update()
                                    type_for_highlight = selected_type
                                    type_update = gr.update()
                                    stop_update = gr.update()
                                    weight_update = gr.update()

                                    try:
                                        from extras.control_hint import (
                                            control_hint_auto_skip_for_selected_type,
                                            control_hint_highlight_style,
                                            detect_control_hint_type_and_default_params,
                                        )
                                    except Exception:
                                        control_hint_auto_skip_for_selected_type = None
                                        control_hint_highlight_style = None
                                        detect_control_hint_type_and_default_params = None

                                    if enabled and detect_control_hint_type_and_default_params is not None:
                                        try:
                                            detected, stop, weight = detect_control_hint_type_and_default_params(image_np)
                                        except Exception:
                                            detected, stop, weight = None, None, None
                                        if detected is not None and stop is not None and weight is not None:
                                            type_for_highlight = detected
                                            type_update = detected
                                            stop_update = float(stop)
                                            weight_update = float(weight)

                                    auto_skip = False
                                    if (
                                        control_hint_auto_skip_for_selected_type is not None
                                        and type_for_highlight is not None
                                    ):
                                        try:
                                            auto_skip, _ = control_hint_auto_skip_for_selected_type(image_np, type_for_highlight)
                                        except Exception:
                                            auto_skip = False

                                    style = ''
                                    if auto_skip and control_hint_highlight_style is not None:
                                        style = control_hint_highlight_style(elem_id)

                                    return style, type_update, stop_update, weight_update
                                return _fn

                            def _ip_make_highlight_fn(elem_id: str):
                                def _fn(image_np, selected_type):
                                    if image_np is None or selected_type is None:
                                        return ''
                                    try:
                                        from extras.control_hint import control_hint_auto_skip_for_selected_type, control_hint_highlight_style
                                        auto_skip, _ = control_hint_auto_skip_for_selected_type(image_np, selected_type)
                                    except Exception:
                                        auto_skip = False
                                        control_hint_highlight_style = None
                                    if auto_skip and control_hint_highlight_style is not None:
                                        return control_hint_highlight_style(elem_id)
                                    return ''
                                return _fn

                            def _ip_auto_detect_all(*args):
                                enabled = args[-1]
                                n = (len(args) - 1) // 2
                                images = args[:n]
                                types_in = args[n:2 * n]
                                styles = []
                                type_updates = []
                                stop_updates = []
                                weight_updates = []

                                try:
                                    from extras.control_hint import (
                                        control_hint_auto_skip_for_selected_type,
                                        control_hint_highlight_style,
                                        detect_control_hint_type_and_default_params,
                                    )
                                except Exception:
                                    control_hint_auto_skip_for_selected_type = None
                                    control_hint_highlight_style = None
                                    detect_control_hint_type_and_default_params = None

                                for idx, (image_np, selected_type) in enumerate(zip(images, types_in)):
                                    elem_id = ip_image_elem_ids[idx] if idx < len(ip_image_elem_ids) else None
                                    type_for_highlight = selected_type
                                    type_update = gr.update()
                                    stop_update = gr.update()
                                    weight_update = gr.update()

                                    if enabled and image_np is not None and detect_control_hint_type_and_default_params is not None:
                                        try:
                                            detected, stop, weight = detect_control_hint_type_and_default_params(image_np)
                                        except Exception:
                                            detected, stop, weight = None, None, None
                                        if detected is not None and stop is not None and weight is not None:
                                            type_for_highlight = detected
                                            type_update = detected
                                            stop_update = float(stop)
                                            weight_update = float(weight)

                                    auto_skip = False
                                    if (
                                        image_np is not None
                                        and control_hint_auto_skip_for_selected_type is not None
                                        and type_for_highlight is not None
                                    ):
                                        try:
                                            auto_skip, _ = control_hint_auto_skip_for_selected_type(image_np, type_for_highlight)
                                        except Exception:
                                            auto_skip = False

                                    style = ''
                                    if auto_skip and control_hint_highlight_style is not None and elem_id is not None:
                                        style = control_hint_highlight_style(elem_id)

                                    styles.append(style)
                                    type_updates.append(type_update)
                                    stop_updates.append(stop_update)
                                    weight_updates.append(weight_update)

                                return styles + type_updates + stop_updates + weight_updates

                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image_elem_id = f'ip_image_{image_count}'
                                    ip_image_elem_ids.append(ip_image_elem_id)
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', image_mode='RGBA', show_label=False, height=300, value=modules.config.default_ip_images[image_count], elem_id=ip_image_elem_id)
                                    ip_detect_style = gr.HTML(value='', elem_classes=['ip_detect_style'])
                                    ip_detect_style_nodes.append(ip_detect_style)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.05, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)
                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.05, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)
                                            filtered_ip_list = [flags.cn_canny, flags.cn_cpds, flags.cn_pose]
                                            default_ip_type = modules.config.default_ip_types[image_count]
                                            if default_ip_type not in filtered_ip_list:
                                                default_ip_type = filtered_ip_list[0]
                                        ip_type = gr.Radio(label='Type', choices=filtered_ip_list, value=default_ip_type, container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)
                                    ip_type.change(lambda x: flags.default_parameters[x] if x in filtered_ip_list else flags.default_parameters[filtered_ip_list[0]],
                                                 inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False) \
                                           .then(fn=_ip_make_highlight_fn(ip_image_elem_id), inputs=[ip_image, ip_type], outputs=[ip_detect_style], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)

                                    ip_image.change(
                                        fn=_ip_make_auto_detect_fn(ip_image_elem_id),
                                        inputs=[ip_image, ip_type, ip_auto_detect],
                                        outputs=[ip_detect_style, ip_type, ip_stop, ip_weight],
                                        queue=False,
                                        show_progress=False
                                    )

                            ip_auto_detect.change(
                                fn=_ip_auto_detect_all,
                                inputs=ip_images + ip_types + [ip_auto_detect],
                                outputs=ip_detect_style_nodes + ip_types + ip_stops + ip_weights,
                                queue=False,
                                show_progress=False
                            )


                        gr.HTML('* Powered by Fooocus Image Mixture Engine (v1.0.1), <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>, and Comfyd workflow engine from ComfyUI.')

                        def ip_advance_checked(x):
                            filtered_ip_list = [flags.cn_canny, flags.cn_cpds, flags.cn_pose]
                            default_ip = filtered_ip_list[0]

                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [default_ip] * len(ip_types) + \
                                [flags.default_parameters[default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Upscale or Variation', id='uov_tab', elem_id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Image', source='upload', type='numpy', image_mode='RGBA', height=300, show_label=False)
                                uov_input_image_full = gr.State(None)
                                with gr.Row():
                                    describe_uov_button = gr.Button(value='Describe Image', variant='secondary', size='sm', visible=False)
                            with gr.Column():
                                with gr.Group():
                                    mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale', value=False)
                                    uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                    with gr.Row():
                                        uov_image_size = gr.Textbox(label='OriginalSize | FinalSize', elem_classes='uov_image_size')
                                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               visible=False, minimum=0, maximum=1.0, step=0.05,
                                                               value=modules.config.default_overwrite_upscale)
                                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            visible=False, minimum=0, maximum=1.0, step=0.05, value=-1)

                                    with gr.Row(visible=False) as uov_hires_fix:
                                        hires_fix_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.05, value=0.8, min_width=20)
                                        hires_fix_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.05, value=0.5, min_width=20)
                                        hires_fix_blurred = gr.Slider(label='Blurred', minimum=0.0, maximum=1.0, step=0.05, value=0.0, min_width=20)
                        uov_input_image.upload(stash_preview_image, inputs=[uov_input_image], outputs=[uov_input_image, uov_input_image_full], show_progress=False, queue=False) \
                            .then(topbar.update_upscale_size_of_image, inputs=[uov_input_image_full, uov_method], outputs=uov_image_size, show_progress=False, queue=False)
                        uov_method.change(topbar.update_size_and_hires_fix, inputs=[uov_input_image_full, uov_method, params_backend, hires_fix_stop, hires_fix_weight, hires_fix_blurred], outputs=[uov_image_size, uov_hires_fix, overwrite_vary_strength, overwrite_upscale_strength], show_progress=False, queue=False)
                        hires_fix_stop.change(lambda x,y,z: sync_backend_params('hires_fix_s',x,y,z), inputs=[hires_fix_stop, params_backend, state_topbar])
                        hires_fix_weight.change(lambda x,y,z: sync_backend_params('hires_fix_w',x,y,z), inputs=[hires_fix_weight, params_backend, state_topbar])
                        hires_fix_blurred.change(lambda x,y,z: sync_backend_params('hires_fix_blurred',x,y,z), inputs=[hires_fix_blurred, params_backend, state_topbar])
                        gr.HTML('* Powered by Fooocus upscale engine, <a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>, and Comfyd workflow engine from ComfyUI.')
                    
                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab', elem_id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint', value=False, container=False)
                            inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox, container=False)
                            invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox, container=False)
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', image_mode='RGBA', tool='sketch', height=350, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_input_image_full = gr.State(None)
                                inpaint_input_image_backend = gr.State(None)
                                with gr.Row():
                                    describe_inpaint_button = gr.Button(value='Describe Image', variant='secondary', size='sm', visible=False)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', show_label=True, source='upload', type='numpy', tool='sketch', height=350, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                inpaint_mask_image_full = gr.State(None)
                                inpaint_mask_image_backend = gr.State(None)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')
                        inpaint_input_image.upload(stash_preview_sketch, inputs=[inpaint_input_image], outputs=[inpaint_input_image, inpaint_input_image_full], show_progress=False, queue=False)
                        inpaint_input_image.clear(lambda: (None, None, None), outputs=[inpaint_input_image_full, inpaint_input_image_backend, inpaint_mask_image_backend], show_progress=False, queue=False)
                        inpaint_mask_image.clear(lambda: None, outputs=[inpaint_mask_image_backend], show_progress=False, queue=False)
                        with gr.Row():
                            inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                            inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.01, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        gr.HTML('* Powered by Fooocus Inpaint Engine, <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>, and Comfyd workflow engine from ComfyUI.')
                        
                        def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                            from extras.inpaint_mask import generate_mask_from_image

                            extras = {}
                            sam_options = None
                            if mask_model == 'u2net_cloth_seg':
                                extras['cloth_category'] = cloth_category
                            elif mask_model == 'sam':
                                sam_options = SAMOptions(
                                    dino_prompt=translator.convert(dino_prompt_text, ads.get_admin_default('translation_methods')),
                                    dino_box_threshold=box_threshold,
                                    dino_text_threshold=text_threshold,
                                    dino_erode_or_dilate=dino_erode_or_dilate,
                                    dino_debug=dino_debug,
                                    max_detections=sam_max_detections,
                                    model_type=sam_model
                                )

                            mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                            return mask


                        inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)
                    with gr.Column(visible=False):
                        with gr.Tab(label='Layer_iclight', id='layer_tab') as layer_tab:
                            with gr.Row():
                                layer_method = gr.Radio(choices=comfy_task.default_method_names, value=comfy_task.default_method_names[0], interactive=False, container=False)
                            with gr.Row():
                                with gr.Column():
                                    layer_input_image = grh.Image(label='Drag given image to here', source='upload', type='numpy', image_mode='RGBA', visible=True, interactive=False)
                                    layer_input_image_full = gr.State(None)
                                with gr.Column():
                                    with gr.Group():
                                        iclight_enable = gr.Checkbox(label='Enable IC-Light', value=True)
                                        iclight_source_radio = gr.Radio(show_label=False, choices=comfy_task.iclight_source_names, value=comfy_task.iclight_source_names[0], elem_classes='iclight_source', elem_id='iclight_source')
                                    gr.HTML('* The module derived from <a href="https://github.com/lllyasviel/IC-Light" target="_blank">IC-Light</a> <a href="https://github.com/layerdiffusion/LayerDiffuse" target="_blank">LayerDiffuse</a>')
                            with gr.Row():
                                example_quick_subjects = gr.Dataset(samples=comfy_task.quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
                            with gr.Row():
                                example_quick_prompts = gr.Dataset(samples=comfy_task.quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
                        example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
                        example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)
                        layer_input_image.upload(stash_preview_image, inputs=[layer_input_image], outputs=[layer_input_image, layer_input_image_full], show_progress=False, queue=False)
                        layer_input_image.change(lambda img, full: None if img is None else full, inputs=[layer_input_image, layer_input_image_full], outputs=[layer_input_image_full], show_progress=False, queue=False)

                    with gr.Tab(label='Enhance+', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False)
                                enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy', image_mode='RGBA')
                                enhance_input_image_full = gr.State(None)
                                with gr.Row():
                                    describe_enhance_button = gr.Button(value='Describe Image', variant='secondary', size='sm', visible=False)
                                with gr.Group():
                                    with gr.Row():
                                        enhance_enabled_1 = gr.Checkbox(label='Enable Region#1', value=False, elem_classes='min_check')
                                        enhance_enabled_2 = gr.Checkbox(label='Enable Region#2', value=False, elem_classes='min_check')
                                        enhance_enabled_3 = gr.Checkbox(label='Enable Region#3', value=False, elem_classes='min_check')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                            with gr.Column():
                                with gr.Row(visible=True) as enhance_input_panel:
                                    with gr.Tabs():
                                        with gr.Tab(label='Upscale  or  Variation'):
                                            with gr.Row():
                                                with gr.Column():
                                                    enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                                        value=modules.config.default_enhance_uov_method)
                                                    enhance_uov_strength = gr.Slider(label='Denoising Strength of enhance',
                                                                        visible=False, minimum=0, maximum=1.0, step=0.01, value=0)
                                                    enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                                    enhance_uov_prompt_type = gr.Radio(label='Prompt.',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)
                                                    
                                                    enhance_uov_method.change(lambda x: gr.update(visible=x.lower() != 'disabled', value=flags.enhance_uov_strengths[x]), inputs=enhance_uov_method, outputs=enhance_uov_strength, queue=False, show_progress=False)
                                                    enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                                        enhance_ctrls = []
                                        enhance_inpaint_mode_ctrls = []
                                        enhance_inpaint_engine_ctrls = []
                                        enhance_inpaint_update_ctrls = []
                                        for index in range(modules.config.default_enhance_tabs):
                                            with gr.Tab(label=f'Region#{index + 1}') as enhance_tab_item:
                                                # enhance_enabled = gr.Checkbox(label='Enable', value=False, 
                                                #         elem_classes='min_check', container=False)
                                                enhance_enabled = [enhance_enabled_1, enhance_enabled_2, enhance_enabled_3][index]
                                                enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       value = 'face' if index==0 else 'hand' if index==1 else 'eye' if index==2 else '',
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                                example_enhance_mask_dino_prompt_text = gr.Dataset(
                                                    samples=modules.config.example_enhance_detection_prompts,
                                                    label='Detection Prompt Quick List',
                                                    components=[enhance_mask_dino_prompt_text],
                                                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                                example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                                                enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                                                enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                                                with gr.Accordion("Detection", open=False):
                                                    enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                                    enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                                    with gr.Accordion("SAM Options",
                                                                    visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                                    open=False) as sam_options:
                                                        enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                                        enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                                        enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                                        enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                                                with gr.Accordion("Inpaint", visible=True, open=False):
                                                    enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method if index not in [0,1,2] else modules.flags.inpaint_option_detail,
                                                                   label='Method', interactive=True)
                                                    enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                                        label='Disable initial latent in inpaint', value=False)
                                                    enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions["z_image_turbo_aio_cn"],
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                                    enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.01,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                                    enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.01,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                                    enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                                    enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                                            enhance_ctrls += [
                                                enhance_enabled,
                                                enhance_mask_dino_prompt_text,
                                                enhance_prompt,
                                                enhance_negative_prompt,
                                                enhance_mask_model,
                                                enhance_mask_cloth_category,
                                                enhance_mask_sam_model,
                                                enhance_mask_text_threshold,
                                                enhance_mask_box_threshold,
                                                enhance_mask_sam_max_detections,
                                                enhance_inpaint_disable_initial_latent,
                                                enhance_inpaint_engine,
                                                enhance_inpaint_strength,
                                                enhance_inpaint_respective_field,
                                                enhance_inpaint_erode_or_dilate,
                                                enhance_mask_invert
                                            ]

                                            enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                                            enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                                            enhance_inpaint_update_ctrls += [[
                                                enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                                enhance_inpaint_strength, enhance_inpaint_respective_field
                                            ]]

                                            enhance_inpaint_mode.change(enhance_inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state, state_topbar], outputs=[
                                                enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                                enhance_inpaint_strength, enhance_inpaint_respective_field
                                            ], show_progress=False, queue=False)

                                            enhance_mask_model.change(
                                                lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                        [gr.update(visible=x == 'sam')] * 2 +
                                                        [gr.Dataset.update(visible=x == 'sam',
                                                            samples=modules.config.example_enhance_detection_prompts)],
                                                inputs=enhance_mask_model,
                                                outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                                        example_enhance_mask_dino_prompt_text],
                                                queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            switch_js_two = "(x,y) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{if(!y){viewer_to_top();}} return [x,y];}"
            down_js = "() => {viewer_to_bottom();}"

            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value=modules.config.default_selected_image_input_tab_id.split('_')[0], visible=False)
            history_link = gr.HTML()

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox, elem_id="scrollable-box-hidden") as advanced_column:
            with gr.Tab(label='Setting', elem_id="scrollable-box") as setting_tab:
                preset_instruction = gr.HTML(visible=False, value=topbar.preset_instruction())
                with gr.Tab(label="General"):
                    performance_selection = gr.Radio(label='Performance', container=False, 
                                                 choices=flags.Performance.list(),
                                                 value=modules.config.default_performance, visible=False)
                    with gr.Group():
                        image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                        with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                            aspect_ratios_selection = gr.Textbox(value='', visible=False, elem_id='aspect_ratios_selection') 
                            with gr.Row():
                                random_aspect_ratio_checkbox = gr.Checkbox(label='Random Aspect Ratio', value=False)
                                use_resolution_override_checkbox = gr.Checkbox(label='Resolution Box', value=False)
                            aspect_ratios_selections = []
                            for template in flags.aspect_ratios_templates:
                                aspect_ratios_selections.append(gr.Radio(label='aspect ratios', choices=flags.available_aspect_ratios_list[template], value=flags.default_aspect_ratios[template], visible= template=='SDXL', info='Vertical(9:16), Portrait(4:5), Photo(4:3), Landscape(3:2), Widescreen(16:9), Cinematic(21:9)', elem_classes='aspect_ratios'))

                            resolution_override = gr.HTML(
                                value="""
                                <div id="resolution_override_widget" style="display:flex; flex-direction:column; gap:12px; padding:12px; border:1px solid var(--neutral-700); border-radius:12px; width:100%; margin:0 auto; align-items:center;">
                                  <div style="display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; width:100%;">
                                    <div style="display:flex; gap:10px; align-items:center; justify-content:flex-start; flex-wrap:wrap;">
                                      <label style="display:flex; gap:6px; align-items:center; font-size:12px; opacity:0.9;">
                                        W
                                        <input data-role="winput" title="图像宽" type="number" min="-1" max="2048" step="1" value="-1" style="width:80px; padding:6px 8px; border-radius:8px; border:1px solid var(--neutral-700); background:var(--neutral-900); color:inherit;" />
                                      </label>
                                      <label style="display:flex; gap:6px; align-items:center; font-size:12px; opacity:0.9;">
                                        H
                                        <input data-role="hinput" title="图像高" type="number" min="-1" max="2048" step="1" value="-1" style="width:80px; padding:6px 8px; border-radius:8px; border:1px solid var(--neutral-700); background:var(--neutral-900); color:inherit;" />
                                      </label>
                                      <select data-role="qstep" title="规格化步长" style="width:32px; padding:6px 6px; border-radius:8px; border:1px solid var(--neutral-700); background:var(--neutral-900); color:inherit; font-size:12px;">
                                        <option value="8" selected>8</option>
                                        <option value="16">16</option>
                                        <option value="32">32</option>
                                        <option value="64">64</option>
                                      </select>
                                    </div>
                                    <div style="display:flex; gap:8px; align-items:center; justify-content:flex-end;">
                                      <button data-role="scale_down" type="button" title="缩小 10%" style="width:36px; height:30px; border-radius:8px; border:1px solid var(--neutral-700); background:var(--neutral-900); color:inherit; cursor:pointer; font-size:14px; line-height:1;">-</button>
                                      <button data-role="scale_up" type="button" title="放大 10%" style="width:36px; height:30px; border-radius:8px; border:1px solid var(--neutral-700); background:var(--neutral-900); color:inherit; cursor:pointer; font-size:14px; line-height:1;">+</button>
                                    </div>
                                  </div>
                                  <div data-role="pad" style="position:relative; width:min(520px, 100%); aspect-ratio:1/1; height:auto; border-radius:12px; border:1px solid var(--neutral-700); background:radial-gradient(circle at 1px 1px, rgba(255,255,255,0.07) 1px, transparent 1px) 0 0 / 18px 18px; overflow:hidden; user-select:none; touch-action:none; margin:0 auto;">
                                    <div data-role="rect" style="position:absolute; left:0; top:0; width:50%; height:50%; background:rgba(255,255,255,0.08); border:2px solid rgba(255,255,255,0.35); border-radius:8px; box-sizing:border-box;">
                                      <div data-role="handle" style="position:absolute; right:0; bottom:0; width:14px; height:14px; border-radius:50%; background:rgba(255,255,255,0.75); border:2px solid rgba(0,0,0,0.35); box-sizing:border-box; transform:translate(50%, 50%);"></div>
                                    </div>
                                  </div>
                                </div>
                                """
                                , visible=False
                            )
                            resolution_quantize_step = gr.Number(value=8, visible=False, elem_id="resolution_quantize_step")
                            overwrite_width = gr.Slider(
                                label='Forced Overwrite of Generating Width',
                                minimum=-1, maximum=2048, step=1, value=-1,
                                visible=False,
                                elem_id="overwrite_width",
                                info='Set as -1 to disable. For developer debugging. Results will be worse for non-standard numbers that SDXL is not trained on.'
                            )
                            overwrite_height = gr.Slider(
                                label='Forced Overwrite of Generating Height',
                                minimum=-1, maximum=2048, step=1, value=-1,
                                visible=False,
                                elem_id="overwrite_height",
                            )

                            def select_random_aspect_ratio(use_random, cached_ratio, current_template='SDXL'):
                                if not use_random:
                                    return [gr.update(), gr.update(), gr.update(), None]

                                if cached_ratio is not None and str(cached_ratio).strip():
                                    try:
                                        import re
                                        raw = str(cached_ratio).split(',', 1)[0]
                                        m2 = re.search(r'(\d+)\D+(\d+)', raw.replace('×', 'x'))
                                        if m2:
                                            width = int(m2.group(1))
                                            height = int(m2.group(2))
                                            return [width, height, cached_ratio, cached_ratio]
                                    except Exception:
                                        pass

                                available_ratios = flags.available_aspect_ratios_list[current_template]
                                if available_ratios:
                                    selected_ratio = random.choice(available_ratios)
                                    try:
                                        import re
                                        raw = str(selected_ratio).split(',', 1)[0]
                                        m2 = re.search(r'(\d+)\D+(\d+)', raw.replace('×', 'x'))
                                        if m2:
                                            width = int(m2.group(1))
                                            height = int(m2.group(2))
                                            return [width, height, selected_ratio, selected_ratio]
                                    except Exception:
                                        pass

                                    width_height = selected_ratio.split('×')[0]
                                    width = int(width_height.split('|')[0] if '|' in width_height else width_height)
                                    for ratio in flags.available_aspect_ratios[flags.aspect_ratios_templates.index(current_template)]:
                                        if str(width) in ratio.split('*')[0]:
                                            height = int(ratio.split('*')[1])
                                            return [width, height, selected_ratio, selected_ratio]

                                return [gr.update(), gr.update(), gr.update(), None]
                            last_preset_ratio = None
                            last_condition_met = False

                            def save_selected_preset_ratio(ratio_value):
                                global last_condition_met, last_preset_ratio
                                if ratio_value and ratio_value.strip() and not ('*' in ratio_value and '×' in ratio_value):
                                    last_preset_ratio = ratio_value
                                return ratio_value

                            for i, aspect_ratios_select in enumerate(aspect_ratios_selections):
                                template = flags.aspect_ratios_templates[i]
                                aspect_ratios_select.change(lambda x, t=template: save_selected_preset_ratio(f"{x},{t}"), inputs=aspect_ratios_select, outputs=aspect_ratios_selection, queue=False, show_progress=False) \
                                    .then(lambda x: None, inputs=aspect_ratios_select, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}') \
                                    .then(lambda: [gr.update(value=-1), gr.update(value=-1)], outputs=[overwrite_width, overwrite_height], queue=False, show_progress=False)

                            def overwrite_aspect_ratios(width, height):
                                global last_condition_met, last_preset_ratio
                                current_condition_met = width > 0 and height > 0
                                last_condition_met = current_condition_met

                                if current_condition_met:
                                    return flags.add_ratio(f'{width}*{height}')
                                else:
                                    return last_preset_ratio if last_preset_ratio is not None else gr.update()

                            overwrite_width.change(overwrite_aspect_ratios, inputs=[overwrite_width, overwrite_height], outputs=aspect_ratios_selection, queue=False, show_progress=False).then(lambda x: x, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                            overwrite_height.change(overwrite_aspect_ratios, inputs=[overwrite_width, overwrite_height], outputs=aspect_ratios_selection, queue=False, show_progress=False).then(lambda x: x, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                        with gr.Row():
                            resolution_multiplier = gr.Slider(label='Resolution Multiply', minimum=1.0, maximum=2.0, step=0.1, value=1.0, elem_id='resolution_multiplier')
                            quick_enhance = gr.Checkbox(label='Quick Enhance', value=False)
                        quick_enhance_uov_strength = gr.Slider(label='Denoising Strength of enhance',
                                         visible=False, minimum=0, maximum=1.0, step=0.01, value=0.2)
                        output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                        negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.", lines=2,
                                             elem_id='negative_prompt', value=modules.config.default_prompt_negative, visible=False)
                        seed_random = gr.Checkbox(label='Random', value=True)
                        image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                    def reversed_checked(r):
                        return gr.update(visible=not r)

                    seed_random.change(reversed_checked, inputs=[seed_random], outputs=[image_seed], queue=False, show_progress=False)
                    scene_seed_random.change(lambda x: [gr.update(value=x), gr.update(visible=not x)], inputs=scene_seed_random, outputs=[seed_random, image_seed], queue=False, show_progress=False)
                    seed_random.change(lambda x: [gr.update(value=x), gr.update(visible=not x)], inputs=seed_random, outputs=[scene_seed_random, scene_image_seed], queue=False, show_progress=False)
                    scene_image_seed.change(lambda x: gr.update(value=x), inputs=scene_image_seed, outputs=image_seed, queue=False, show_progress=False)
                    image_seed.change(lambda x: gr.update(value=x), inputs=image_seed, outputs=scene_image_seed, queue=False, show_progress=False)
                    quick_enhance.change(fn=lambda x: [x, 'Upscale (1.5x)' if x else 'Disabled', gr.update(visible=x, value=0.2)],
                                        inputs=quick_enhance,outputs=[enhance_checkbox, enhance_uov_method, quick_enhance_uov_strength], queue=False, show_progress=False)
                    quick_enhance_uov_strength.change(fn=lambda x: x,inputs=quick_enhance_uov_strength,outputs=enhance_uov_strength, queue=False, show_progress=False)

                with gr.Tab(label="Advanced"):
                    with gr.Group():
                        guidance_scale = gr.Slider(label='Guidance Scale', minimum=0.01, maximum=30.0, step=0.01,
                                            value=modules.config.default_cfg_scale,
                                            info='Higher value means style is cleaner, vivider, and more artistic.')
                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        with gr.Row():
                            sampler_name = gr.Dropdown(label='Sampler', choices=flags.comfy_sampler_list,
                                                   value=modules.config.default_sampler)
                            scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.comfy_scheduler_list,
                                                     value=modules.config.default_scheduler)
                        with gr.Row():
                            vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)
                            clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip)
                    sdxl_adv_checkbox = gr.Checkbox(label='SDXL advanced setting', value=False,  container=False)
                    with gr.Group(visible=False) as sdxl_adv_pannel: 
                        with gr.Row():
                            sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.01,
                                      value=modules.config.default_sample_sharpness)
                            adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr)
                        with gr.Row():
                            refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])
                            overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch)
                        with gr.Row():
                            adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0, step=0.01, value=1.5)
                            adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0, step=0.01, value=0.8)
                            adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                    def toggle_checked(r):
                        return gr.update(visible=r)

                    sdxl_adv_checkbox.change(toggle_checked, inputs=[sdxl_adv_checkbox], outputs=[sdxl_adv_pannel], queue=False, show_progress=False)

                with gr.Tab(label='Control'):
                    debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                    skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')
                    controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.01, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')
                    canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                    canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)
                with gr.Tab(label='Inpaint'):
                    with gr.Group():
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                    value=modules.config.default_inpaint_engine_version,
                                    choices=flags.inpaint_engine_versions["z_image_turbo_aio_cn"])
                        with gr.Row():
                            debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                            inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)    
                        with gr.Row():
                            debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False)
                            debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False)
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')
                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                    inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                    inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                    inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                with gr.Tabs():
                    with gr.Tab(label='Describe Image', id='describe_tab', visible=True) as image_describe:
                        with gr.Group():
                            with gr.Column():
                                describe_input_image = grh.Image(label='Image to be described', source='upload', type='numpy', show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    describe_methods = gr.CheckboxGroup(
                                        label='Content Type', 
                                        choices=flags.describe_types,
                                        value=modules.config.default_describe_content_type, visible= not MiniCPM.get_enable(),
                                        info="To use Agent features, go to ↗ 'Identity' settings to enable VLM")
                                    describe_prompt = gr.Textbox(label="VLM enabled: Enter additional prompts (optional).", show_label=True, lines=1, max_lines=10, placeholder="Type additional prompt for describe image.", visible=MiniCPM.get_enable())
                                    with gr.Row():
                                        describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox, visible=not MiniCPM.get_enable())
                                        describe_output_tags = gr.Checkbox(label='Output with tags', value=False, visible=MiniCPM.get_enable(), min_width=50)
                                        describe_output_chinese = gr.Checkbox(label='Output in Chinese', value=False, visible=MiniCPM.get_enable(), min_width=50)
                                        describe_output_artist = gr.Checkbox(label='Artist', value=False, visible=MiniCPM.get_enable(), min_width=50)
                                describe_image_size = gr.Button(label='Original Size / Recommended Size', elem_id='describe_image_size', visible=False)
                                with gr.Row():
                                    describe_btn = gr.Button(value='⚡ Execute Instruction' if MiniCPM.get_enable() else 'Describe this Image into Prompt')
                                    unload_btn = gr.Button(value='🗑️Unload Models', min_width=150)
                                with gr.Column(visible=MiniCPM.get_enable()) as vlm_describe_col:
                                    vlm_status_info = gr.HTML(value=f'<div style="margin-bottom: 5px;"> 🤖 <b>VLM Model:</b> <span style="color: #2196F3;">{MiniCPM.current_version}</span></div>')

                                def trigger_show_image_properties(image):
                                    image_size = modules.util.get_image_size_info(image, modules.flags.available_aspect_ratios[0])
                                    return gr.update(value=image_size, visible=True)

                                def apply_recommended_size(button_text):
                                    try:
                                        if ' / ' in button_text:
                                            parts = button_text.split(' / ')
                                            if len(parts) > 1:
                                                recommended_part = parts[1].strip()
                                            else:
                                                recommended_part = button_text
                                        else:
                                            recommended_part = button_text

                                        if 'x' in recommended_part:
                                            size_part = recommended_part.split('|')[0].strip()
                                            width_str, height_str = size_part.split('x')[:2]
                                            width = int(''.join(filter(str.isdigit, width_str)))
                                            height = int(''.join(filter(str.isdigit, height_str)))
                                            return [width, height]
                                        else:
                                            return [gr.update(), gr.update()]
                                    except Exception:
                                        return [gr.update(), gr.update()]

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image, outputs=describe_image_size, show_progress=False, queue=False)
                                describe_image_size.click(apply_recommended_size, inputs=describe_image_size, outputs=[overwrite_width, overwrite_height], show_progress=False, queue=False)

                    with gr.Tab(label='Metadata', id='metadata_tab', visible=True) as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='Drag any image generated by Fooocus here', source='upload', type='pil')
                            with gr.Accordion("Preview Metadata", open=True, visible=True) as metadata_preview:
                                metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata', interactive=False)

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return [results, gr.update(interactive=parameters is not None)]

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                outputs=[metadata_json, metadata_import_button], queue=False, show_progress=True)
                    import enhanced.image_encrypt_tab as image_encrypt_tab
                    image_encrypt_tab.add_image_encrypt_tab(progress_window, progress_gallery, gallery, progress_video, comparison_box, compare_btn, comparison_state, state_topbar, state_is_generating, image_toolbox, gallery_index, output_format)
                    # custom plugin "OneButtonPrompt"
                    with gr.Tab(label="OneButtonPrompt"):
                        import custom.OneButtonPrompt.ui_onebutton as ui_onebutton
                        run_event = gr.Number(visible=False, value=0)
                        ui_onebutton.ui_onebutton(prompt, run_event, random_button)
                        super_prompter_prompt = gr.Textbox(label='Prompt prefix', value='Expand the following prompt to add more detail:', lines=1)
                    
            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)
                with gr.Row():
                    with gr.Column(scale=5):
                        style_search_bar = gr.Textbox(show_label=False, container=False,
                                                    placeholder="\U0001F50E Type here to search styles ...",
                                                    value="",
                                                    label='Search Styles',
                                                    scale=5)

                style_selections = gr.CheckboxGroup(show_label=False, container=False, visible=True,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                def generate_style_grid_html():
                    import json
                    html = ""
                    for style in legal_style_names:
                        style_data = modules.sdxl_styles.get_style_config(style)
                        is_primary = style in modules.config.default_styles
                        variant_class = "primary" if is_primary else "secondary"
                        style_data_json = json.dumps(style_data).replace('"', '&quot;')
                        html += f"""
                        <div class="style_item" data-style-data="{style_data_json}">
                            <button class="style-button {variant_class}" data-style-name="{style}">{style}</button>
                        </div>
                        """
                    return html

                with gr.Column(visible=False, elem_id="style_visual_layout_container", elem_classes=["scrollable-box"]) as visual_layout_container:
                    style_grid_html = gr.HTML(value=generate_style_grid_html(), elem_classes=["style_grid"], elem_id="style_grid")

                has_loaded = gr.State(value=0)

                def toggle_layout(use_visual, current_loaded):
                                  new_loaded = 0 if use_visual and current_loaded == 0 else current_loaded
                                  return [gr.update(visible=not use_visual),gr.update(visible=use_visual),new_loaded]

                style_selections.change(fn=None,
                                        inputs=style_selections,
                                        outputs=None,
                                        _js='() => { refresh_style_layout(); }')

                prompt.change(lambda x,y: calculateTokenCounter(x,y), inputs=[prompt, style_selections], outputs=prompt_token_counter)

            with gr.Tab(label='Models', elem_id="scrollable-box"):
                gallery_visible = gr.State(value=False)
                current_previews = gr.State(value=[])
                active_target = gr.State(value="base")
                lora_gallery_visible = [gr.State(value=False) for _ in range(len(modules.config.default_loras))]
                lora_current_previews = [gr.State(value=[]) for _ in range(len(modules.config.default_loras))]
                script_dir = os.path.dirname(os.path.abspath(__file__))

                def load_config_paths():
                    config_path = os.path.normpath(os.path.join(script_dir, "..", "..", "users", "config.txt"))
                    default_paths = { "path_checkpoints": [], "path_loras": [] }
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        if isinstance(config, list) and len(config) > 0:
                            config = config[0]
                        config["path_checkpoints"] = [
                            os.path.normpath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                            for p in config.get("path_checkpoints", default_paths["path_checkpoints"])]
                        config["path_loras"] = [
                            os.path.normpath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                            for p in config.get("path_loras", default_paths["path_loras"])]
                        return config
                    except Exception as e:
                        print(f"Error loading config: {e}, using default paths")
                        return default_paths

                def get_model_previews():
                    previews = []
                    no_image_path = os.path.normpath(os.path.join(script_dir, "presets", "samples", "noimage.jpg"))
                    for model_name in modules.config.model_filenames:
                        model_key = str(model_name).replace("\\", "/").lstrip("/")
                        model_full_path = None
                        for catalog in ("checkpoints", "diffusion_models"):
                            try:
                                resolved = modules.config.modelsinfo.get_model_filepath(catalog, model_key)
                            except Exception:
                                resolved = ""
                            if resolved:
                                model_full_path = os.path.normpath(resolved)
                                break
                        if not model_full_path:
                            continue

                        root = os.path.dirname(model_full_path)
                        model_file = os.path.basename(model_full_path)
                        base_name = os.path.splitext(model_file)[0]

                        image_path = None
                        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                            possible_path = os.path.normpath(os.path.join(root, f"{base_name}{ext}"))
                            if os.path.exists(possible_path):
                                image_path = possible_path
                                break
                        if not image_path:
                            parent_dir = os.path.dirname(root)
                            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                possible_path = os.path.normpath(os.path.join(parent_dir, f"{base_name}{ext}"))
                                if os.path.exists(possible_path):
                                    image_path = possible_path
                                    break
                        image_path = image_path or no_image_path
                        display_name = str(model_name).replace('/', '\\')
                        previews.append((image_path, display_name, model_full_path))
                    return previews

                def get_lora_previews():
                    config = load_config_paths()
                    allowed_loras = {
                        os.path.normpath(path).lower().replace('/', '\\')
                        for path in modules.config.lora_filenames
                    }
                    previews = []
                    for lora_dir in config.get("path_loras", []):
                        if not os.path.exists(lora_dir):
                            continue
                        for root, dirs, files in os.walk(lora_dir):
                            lora_files = [f for f in files if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.gguf'))]
                            for lora_file in lora_files:
                                full_path = os.path.normpath(os.path.join(root, lora_file))
                                relative_lora_path = os.path.relpath(full_path, lora_dir).replace('/', '\\').lower()
                                filename_only = lora_file.lower()
                                if filename_only not in allowed_loras and relative_lora_path not in allowed_loras:
                                    continue
                                relative_path = os.path.relpath(root, lora_dir).replace('/', '\\')
                                base_name = os.path.splitext(lora_file)[0]
                                lora_full_path = os.path.normpath(os.path.join(root, lora_file))
                                image_path = None
                                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                    possible_path = os.path.normpath(os.path.join(root, f"{base_name}{ext}"))
                                    if os.path.exists(possible_path):
                                        image_path = possible_path
                                        break
                                if not image_path and relative_path != ".":
                                    parent_dir = os.path.dirname(root)
                                    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                        possible_path = os.path.normpath(os.path.join(parent_dir, f"{base_name}{ext}"))
                                        if os.path.exists(possible_path):
                                            image_path = possible_path
                                            break
                                image_path = image_path or os.path.normpath(os.path.join(script_dir, "presets", "samples", "noimage.jpg"))
                                display_name = f"{relative_path}\{lora_file}" if relative_path != "." else lora_file
                                previews.append((image_path, display_name, lora_full_path))
                    return previews

                def show_model_gallery(current_visible, current_active_target, target_type, state_params, use_model_filter):
                                       refresh_files_clicked(state_params, use_model_filter, False)
                                       if current_active_target != target_type:
                                           new_visible = True
                                       else:
                                           new_visible = not current_visible
                                       if new_visible:
                                           previews = get_model_previews()
                                           return [gr.Gallery.update(value=[(p[0], p[1]) for p in previews], visible=new_visible),new_visible,previews,target_type,
                                                   gr.Button.update(variant="primary" if target_type == "base" else "secondary"),
                                                   gr.Button.update(variant="primary" if target_type == "refiner" else "secondary")]
                                       else:
                                           return [gr.Gallery.update(visible=False, value=[]),new_visible,[],current_active_target,
                                                   gr.Button.update(variant="secondary"),
                                                   gr.Button.update(variant="secondary")]

                def show_lora_gallery(current_visible, index, state_params, use_model_filter):
                                      new_visible = not current_visible
                                      refresh_files_clicked(state_params, use_model_filter, False)
                                      previews = get_lora_previews()
                                      lora_current_previews[index].value = previews
                                      return (gr.Gallery.update(value=[(p[0], p[1]) for p in previews], visible=new_visible), new_visible, previews, gr.Button.update(variant="primary" if new_visible else "secondary"))

                def on_gallery_select(evt: gr.SelectData, previews, target_type):
                                      if previews and evt.index < len(previews):
                                          selected_path = previews[evt.index][1].replace('/', '\\')
                                          if target_type == "base":
                                              return [gr.Dropdown.update(value=selected_path), gr.Dropdown.update(), gr.Gallery.update()]
                                          elif target_type == "refiner":
                                              return [gr.Dropdown.update(), gr.Dropdown.update(value=selected_path), gr.Gallery.update()]
                                      return [gr.Dropdown(), gr.Dropdown(), gr.Gallery.update()]

                def on_lora_gallery_select(evt: gr.SelectData, previews, index):
                                           if previews and isinstance(previews, list):
                                               if evt.index < len(previews):
                                                   selected_path = previews[evt.index][1].replace('/', '\\')
                                                   return gr.Dropdown.update(value=selected_path)
                                           return gr.Dropdown.update(value='None')

                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (or HighNoise)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True,
                                                 elem_id="model_dropdown_base",elem_classes="model-dropdown",interactive=True, info="Right Click for Model Gallery")
                        refiner_model = gr.Dropdown(label='Refiner (or LowNoise)', choices=['None'] + modules.config.get_base_model_list('Fooocus', None), value=modules.config.default_refiner_model_name, show_label=True,
                                                 elem_id="model_dropdown_refiner",elem_classes="model-dropdown",interactive=True, info="WanT2I selects VACE here", visible=False)
                    with gr.Row():
                        base_preview_btn = gr.Button( "🖼️ Base Model", variant="secondary", visible=False,elem_id="base_preview_btn")
                        refiner_preview_btn = gr.Button("🖼️ Refiner", variant="secondary", visible=False,elem_id="refiner_preview_btn")
                    model_gallery = gr.Gallery(label="Model Previews", columns=4, rows=2, height="auto", visible=False, elem_classes="model-gallery")
                    base_preview_btn.click(fn=lambda cv, cat, tt, sp, umf: show_model_gallery(cv, cat, tt, sp, umf),
                                           inputs=[gallery_visible, active_target, gr.State("base"), state_topbar, model_filter_state],
                                           outputs=[model_gallery, gallery_visible, current_previews, active_target, base_preview_btn, refiner_preview_btn], show_progress=False, queue=False) \
                                           .then(fn=None,_js='''(galleryVisible, activeTarget) => {highlightModelDropdown("base");}''')
                    refiner_preview_btn.click(fn=lambda cv, cat, tt, sp, umf: show_model_gallery(cv, cat, tt, sp, umf),
                                              inputs=[gallery_visible, active_target, gr.State("refiner"), state_topbar, model_filter_state],
                                              outputs=[model_gallery, gallery_visible, current_previews, active_target, base_preview_btn, refiner_preview_btn], show_progress=False, queue=False) \
                                           .then(fn=None,_js='''(galleryVisible, activeTarget) => {highlightModelDropdown("refiner");}''')
                    model_gallery.select(on_gallery_select, inputs=[current_previews, active_target], outputs=[base_model, refiner_model, model_gallery], show_progress=False, queue=False)
                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Value for switching two models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)
                    scene_base_model.change(fn=lambda x: x, inputs=[scene_base_model], outputs=[base_model], show_progress=False, queue=False)
                    scene_refiner_model.change(
                        fn=lambda x: (x, gr.update(visible=x != 'None')),
                        inputs=[scene_refiner_model],
                        outputs=[refiner_model, refiner_switch],
                        show_progress=False,
                        queue=False,
                    )
                with gr.Group():
                    lora_ctrls = []
                    lora_galleries = []
                    lora_preview_btns = []
                    lora_models = []
                    lora_trigger_words = []
                    lora_send_to_prompt_btns = []
                    lora_save_btns = []
                    with gr.Row():
                        show_trigger_words_panel = gr.Checkbox(label='Show Trigger Words Panel', value=False, elem_classes='show_trigger_words_panel')
                        use_model_filter_checkbox = gr.Checkbox(label='Use Model Filters', value=True, elem_classes='use_model_filter_checkbox')
                    trigger_word_containers = []
                    from modules.lora_trigger_manager import get_lora_trigger_word, update_trigger_word, save_trigger_word, send_trigger_to_prompt

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'],scale=1, min_width=40)
                            lora_preview_btn = gr.Button(f"🖼️", variant="secondary",
                                                         elem_id=f"lora_preview_btn_{i}", visible=False)
                            lora_preview_btns.append(lora_preview_btn)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5, elem_id=f"lora_dropdown_{i}")
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.05, value=weight,
                                                    elem_classes='lora_weight', scale=5)

                            trigger_word_value = get_lora_trigger_word(filename) if filename != 'None' else ''
                        with gr.Row(visible=False) as trigger_container:
                            trigger_word_containers.append(trigger_container)
                            lora_trigger_word = gr.Textbox(label='Trigger Word', value=trigger_word_value,
                                                         placeholder='Input LoRA trigger word',
                                                         elem_id=f"lora_trigger_word_{i}", min_width=300,lines=1, scale=5)
                            lora_trigger_words.append(lora_trigger_word)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]
                            lora_models.append(lora_model)
                            with gr.Column(min_width=80):
                                send_to_prompt_btn = gr.Button("✅", variant="secondary",
                                                            elem_id=f"lora_send_to_prompt_{i}", elem_classes='lora_send_to_prompt')
                                lora_save_btns.append(gr.Button("💾", variant="secondary",
                                                            elem_id=f"lora_save_{i}", elem_classes='lora_save'))
                            lora_save_btns[i].click(
                                fn=save_trigger_word,
                                inputs=[lora_models[i], lora_trigger_words[i]],
                                outputs=[lora_trigger_words[i]]
                            )
                            lora_send_to_prompt_btns.append(send_to_prompt_btn)
                            send_to_prompt_btn.click(
                                fn=send_trigger_to_prompt,
                                inputs=[lora_model, lora_trigger_word],
                                outputs=[lora_send_to_prompt_btns[i]]
                            ).then(fn=None, _js=f"() => {{ if(window.globalAutoAddLoraTriggerWord) window.globalAutoAddLoraTriggerWord('{lora_trigger_word.elem_id}', '{lora_model.elem_id}') }}")
                        with gr.Row():
                            lora_gallery = gr.Gallery(label=f"LoRA {i + 1} Previews", columns=4, rows=2, height="auto", visible=False, elem_classes="lora-gallery")
                            lora_galleries.append(lora_gallery)
                    show_trigger_words_panel.change(
                        fn=lambda visible: [gr.update(visible=visible)] * len(trigger_word_containers),
                        inputs=[show_trigger_words_panel],
                        outputs=trigger_word_containers,
                        queue=False, show_progress=False
                    )
                    for i in range(len(lora_models)):
                        lora_models[i].change(
                            fn=update_trigger_word,
                            inputs=[lora_models[i]],
                            outputs=[lora_trigger_words[i]],
                            queue=False, show_progress=False
                        ).then(fn=None, _js=f"() => {{ if(window.globalAutoAddLoraTriggerWord) window.globalAutoAddLoraTriggerWord('{lora_trigger_words[i].elem_id}', '{lora_models[i].elem_id}') }}")

                    scene_lora_models = [scene_lora_model, scene_lora_model_2, scene_lora_model_3, scene_lora_model_4]
                    scene_lora_weights = [scene_lora_weight, scene_lora_weight_2, scene_lora_weight_3, scene_lora_weight_4]
                    scene_lora_trigger_words = [scene_lora_trigger_words[0], scene_lora_trigger_words[1], scene_lora_trigger_words[2],scene_lora_trigger_words[3]]

                    for i, (scene_model, scene_weight, ctrl_idx, trigger_word_id) in enumerate([
                        (scene_lora_model, scene_lora_weight, 1, scene_lora_trigger_words[0].elem_id if scene_lora_trigger_words else ''),
                        (scene_lora_model_2, scene_lora_weight_2, 4, scene_lora_trigger_words[1].elem_id if scene_lora_trigger_words else ''),
                        (scene_lora_model_3, scene_lora_weight_3, 7, scene_lora_trigger_words[2].elem_id if len(scene_lora_trigger_words)>1 else ''),
                        (scene_lora_model_4, scene_lora_weight_4, 10, scene_lora_trigger_words[3].elem_id if len(scene_lora_trigger_words)>2 else '')
                    ]):
                        scene_model.change(
                            fn=lambda model, weight, lock, idx=i, cidx=ctrl_idx: (gr.update(), gr.update()) if lock else (model, weight),
                            inputs=[scene_model, scene_weight, scene_to_main_sync_lock],
                            outputs=[lora_ctrls[ctrl_idx], lora_ctrls[ctrl_idx+1]],
                            queue=False, show_progress=False
                        ).then(fn=update_trigger_word, inputs=[scene_model], outputs=[scene_lora_trigger_words[i] if scene_lora_trigger_words and i<len(scene_lora_trigger_words) else gr.Textbox()], queue=False, show_progress=False)
                        if trigger_word_id and hasattr(scene_model, 'elem_id'): scene_model.change(fn=update_trigger_word, inputs=[scene_model], outputs=[scene_lora_trigger_words[i] if scene_lora_trigger_words and i<len(scene_lora_trigger_words) else gr.Textbox()], show_progress=False, queue=False)
                        scene_lora_weight.change(fn=lambda weight: weight, inputs=[scene_lora_weight], outputs=[lora_ctrls[2]], show_progress=False, queue=False)
                        scene_lora_weight_2.change(fn=lambda weight: weight, inputs=[scene_lora_weight_2], outputs=[lora_ctrls[5]], show_progress=False, queue=False)
                        scene_lora_weight_3.change(fn=lambda weight: weight, inputs=[scene_lora_weight_3], outputs=[lora_ctrls[8]], show_progress=False, queue=False)
                        scene_lora_weight_4.change(fn=lambda weight: weight, inputs=[scene_lora_weight_4], outputs=[lora_ctrls[11]], show_progress=False, queue=False)
                    for i in range(len(modules.config.default_loras)):
                        lora_galleries[i].select(on_lora_gallery_select,
                                                 inputs=[lora_current_previews[i], gr.State(i)],
                                                 outputs=[lora_models[i]], show_progress=False, queue=False)
                        lora_preview_btns[i].click(fn=lambda current_visible, sp, umf,
                                                   idx=i: show_lora_gallery(current_visible, idx, sp, umf),
                                                   inputs=[lora_gallery_visible[i], state_topbar, model_filter_state],
                                                   outputs=[lora_galleries[i], lora_gallery_visible[i], lora_current_previews[i], lora_preview_btns[i]], show_progress=False, queue=False)
                with gr.Row():
                    refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
                #with gr.Row():
                #    sync_model_info = gr.Checkbox(label='Sync model info', interactive=False, info='Improve usability and transferability for preset and embedinfo.', value=False, container=False)
                #    with gr.Column(visible=False) as info_sync_texts:
                #        models_infos = []
                #info_sync_button = gr.Button(label='Sync', value='\U0001f504 Sync Remote Info', visible=False, variant='secondary', elem_classes='refresh_button')
                #info_progress = gr.Markdown('Note: If MUID is not obtained after synchronizing, it means that it is a new model file. You need to add an available download URL before.', visible=False)
                #sync_model_info.change(lambda x: (gr.update(visible=x), gr.update(visible=x),  gr.update(visible=x)), inputs=sync_model_info, outputs=[info_sync_texts, info_sync_button, info_progress], queue=False, show_progress=False)
                #info_sync_button.click(toolbox.sync_model_info_click, inputs=models_infos, outputs=models_infos, queue=False, show_progress=False)

                refresh_files_output = [base_model, refiner_model, vae_name] + scene_lora_ctrls
                refresh_files_targets = refresh_files_output + lora_ctrls
                refresh_files.click(refresh_files_clicked, [state_topbar, model_filter_state], refresh_files_output + lora_ctrls,
                                    queue=True, show_progress=False)
                scene_refresh_files.click(refresh_files_clicked, [state_topbar, model_filter_state], refresh_files_output + lora_ctrls,
                                    queue=True, show_progress=False)

                def _on_model_filter_toggle_from_main(state_params, use_model_filter, sync_lock, current_state):
                    if sync_lock:
                        return [current_state, gr.update(), False] + [gr.update()] * len(refresh_files_targets)
                    return [use_model_filter, gr.update(value=use_model_filter), True] + refresh_files_clicked(state_params, use_model_filter)

                def _on_model_filter_toggle_from_scene(state_params, use_model_filter, sync_lock, current_state):
                    if sync_lock:
                        return [current_state, gr.update(), False] + [gr.update()] * len(refresh_files_targets)
                    return [use_model_filter, gr.update(value=use_model_filter), True] + refresh_files_clicked(state_params, use_model_filter)

                use_model_filter_checkbox.change(
                    fn=_on_model_filter_toggle_from_main,
                    inputs=[state_topbar, use_model_filter_checkbox, model_filter_sync_lock, model_filter_state],
                    outputs=[model_filter_state, scene_use_model_filter_checkbox, model_filter_sync_lock] + refresh_files_targets,
                    queue=True,
                    show_progress=False,
                )

                scene_use_model_filter_checkbox.change(
                    fn=_on_model_filter_toggle_from_scene,
                    inputs=[state_topbar, scene_use_model_filter_checkbox, model_filter_sync_lock, model_filter_state],
                    outputs=[model_filter_state, use_model_filter_checkbox, model_filter_sync_lock] + refresh_files_targets,
                    queue=True,
                    show_progress=False,
                )

            # with gr.Tab(label='Gallery', elem_id="scrollable-box"):
            #     with gr.Row():
            #         gallery_id_button = gr.Button(value='GalleryCenter(under construction...)', visible=True, elem_id="gallery_center")

            with gr.Tab(label='Identity', elem_id="scrollable-box"):
                binding_id_button = gr.Button(value='IdentityCenter', visible=True, elem_id="identity_center")
                identity_introduce = gr.HTML(visible=True, value=topbar.identity_introduce, elem_classes=["identityIntroduce"], elem_id='identity_introduce')
                with gr.Column() as configure_panel:
                    with gr.Tab(label='Application') as user_panel:
                        with gr.Row():
                            language_ui = gr.Radio(label='Language of UI', choices=['En', '中文'], value=modules.flags.language_radio(args_manager.args.language), interactive=(args_manager.args.language in ['default', 'cn', 'en']), container=False)
                            background_theme = gr.Radio(label='Theme of background', choices=['light', 'dark'], value=args_manager.args.theme, interactive=True, container=False)
                        with gr.Group():
                            mobile_link = gr.HTML(elem_classes=["htmlcontent"], value=f'{get_local_url}/<div>Mobile phone access address within the LAN. If you want WAN access, consulting QQ group: 1005085136.</div>')
                            prompt_preset_button = gr.Button(value='Save the current parameters as a preset package')
                            backfill_prompt = gr.Checkbox(label='Backfill prompt while switching images', value=modules.config.default_backfill_prompt)
                            style_preview_checkbox = gr.Checkbox(label="Enable visual style preview", value=False)
                            disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw)
                            disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance))
                            disable_seed_increment = gr.Checkbox(label='Disable seed increment', value=False)
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image', visible=not args_manager.args.disable_image_log,
                                                                         value=modules.config.default_save_only_final_enhanced_image)
                            read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False, visible=False)
                            no_welcome_checkbox = gr.Checkbox(label="Hide welcome picture", value=False)
                            missing_model_filter_checkbox = gr.Checkbox(label="Missing model filter", value=False, info="Filtering presets with missing models")
                            no_model_modal_checkbox = gr.Checkbox(label="Disable model download notification", value=False)

                        with gr.Group():
                            image_tools_checkbox = gr.Checkbox(label='Enable ParamsTools', value=True, info='Management of published image sets, located in the middle toolbox on the right side of the image set.')
                            generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                        info='(Experimental) This may cause performance problems on some computers and certain internet conditions.', value=False)
                            black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                            black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)
                            save_metadata_to_images.change(toggle_checked, inputs=[save_metadata_to_images], outputs=[metadata_scheme], queue=False, show_progress=False)
                    style_preview_checkbox.change(toggle_layout,
                                    inputs=[style_preview_checkbox, has_loaded],
                                    outputs=[style_selections, visual_layout_container, has_loaded],
                                    queue=False,
                                    show_progress=False) \
                                .then(lambda x,y: ads.set_user_default_value("style_preview_checkbox",x,y), inputs=[style_preview_checkbox, state_topbar]) \
                                .then(lambda: None, _js='()=>{refresh_style_localization(); refresh_style_layout();}')

                    style_search_bar.change(fn=None,
                                        inputs=None,
                                        outputs=None,
                                        _js='()=>{refresh_style_layout();}')

                    gradio_receiver_style_selections.input(fn=None,
                                                       inputs=None,
                                                       outputs=None,
                                                       _js='()=>{refresh_style_layout();}')
                    no_welcome_checkbox.change(
                        lambda x,y: ads.set_admin_default_value("no_welcome_checkbox", x, y),
                        inputs=[no_welcome_checkbox, state_topbar],
                        outputs=None,
                        queue=False
                    ).then(
                        lambda x: gr.update(value=None) if x else gr.update(),
                        inputs=[no_welcome_checkbox],
                        outputs=progress_window
                    )
                    missing_model_filter_checkbox.change(
                        lambda x,y: ads.set_admin_default_value("missing_model_filter_checkbox", x, y),
                        inputs=[missing_model_filter_checkbox, state_topbar],
                        outputs=None,
                        queue=False
                    )
                    no_model_modal_checkbox.change(
                        lambda x, y: ads.set_user_default_value("no_model_modal_checkbox", x, y),
                        inputs=[no_model_modal_checkbox, state_topbar],
                        outputs=None,
                        queue=False
                    )

                    with gr.Tab(label='Local System') as local_system_tab:
                        with gr.Column() as admin_panel:
                            with gr.Group():
                                with gr.Row(visible=True if not args_manager.args.disable_backend else False):
                                    admin_link = gr.HTML(elem_classes=["htmlcontent"])
                                    admin_mgr_link = gr.HTML(value='')
                                with gr.Group():
                                    with gr.Row(visible=True if not args_manager.args.disable_backend else False):
                                        admin_sync_button = gr.Button(value='Sync presets nav to guest', size="sm", min_width=70)
                                    with gr.Row(visible=True if not args_manager.args.disable_backend else False):
                                        comfyd_active_checkbox = gr.Checkbox(label='Enable Comfyd always active', value=ads.get_admin_default('comfyd_active_checkbox') and not args_manager.args.disable_comfyd and not args_manager.args.disable_backend, info='Enabling will improve execution speed.')
                                        fast_comfyd_checkbox = gr.Checkbox(label='Enable optimizations for Comfyd', value=ads.get_admin_default('fast_comfyd_checkbox'), info='Effective for some Nvidia cards.')
                                        cache_clear_on_finish_checkbox = gr.Checkbox(label='Clear caches on finish', value=ads.get_admin_default('cache_clear_on_finish_checkbox'), info='Restart Comfyd. Clear execution caches and unload models after each task.')
                                    with gr.Row():
                                        minicpm_checkbox = gr.Checkbox(label='Enable VLM', value=ads.get_admin_default('minicpm_checkbox'), info='Enable it for describe, translate and expand.')
                                        advanced_logs = gr.Checkbox(label='Enable advanced logs', value=ads.get_admin_default('advanced_logs'), info='Enabling with more infomation in logs.')
                                        with gr.Column():
                                            minicpm_version = gr.Dropdown(label='VLM Version', choices=['Qwen3-VL-2B-Instruct-abliterated', 'Qwen3-VL-4B-Instruct-abliterated', 'Qwen3-VL-8B-Instruct-abliterated', 'Qwen3.5-9B-ultra-heretic', 'MiniCPMv45', 'MiniCPMv26'], value=ads.get_admin_default('minicpm_version'), info='Select the VLM model version to use')
                                    with gr.Column(visible=True if not args_manager.args.disable_backend else False):
                                        reserved_vram = gr.Slider(label='Reserved VRAM(GB)', minimum=0, maximum=24, step=0.1, value=ads.get_admin_default('reserved_vram'), info='Reserve VRAM to prevent OOM or Slow inference.')
                                        cache_ram = gr.Slider(label='Cache RAM(GB)', minimum=0, maximum=96, step=0.1, value=ads.get_admin_default('cache_ram'), info='[BETA]Set RAM cache threshold. 0: Classic; >0: RAM Pressure mode (auto-purge when available RAM is low).')
                                        wavespeed_strength = gr.Slider(label='wavespeed_strength', minimum=0, maximum=1, step=0.01, value=ads.get_admin_default('wavespeed_strength'), info='Wavespeed optimization strength to improve inference speed on some presets.')
                                    with gr.Row(visible=True if not args_manager.args.disable_backend else False):
                                        translation_methods = gr.Radio(label='Translation methods', choices=modules.flags.translation_methods, value=ads.get_admin_default('translation_methods'))

                            with gr.Row(visible=True):
                                with gr.Group():
                                    web_in_did_title = gr.Markdown(value="Accessed Users:", elem_classes=["p2p_title"])
                                    with gr.Row():
                                        web_in_did_input = gr.Textbox(max_lines=1, container=False, placeholder="Type did here.", min_width=60, elem_classes='web_input1')
                                        web_in_did_add_btn = gr.Button(value="Add", size="sm", min_width=30)
                                        web_in_did_del_btn = gr.Button(value="Del", size="sm", min_width=30)
                                        web_in_did_switch_btn = gr.Button(value="Switch", size="sm", min_width=30)
                                    web_in_did_list = gr.Markdown(elem_classes=["htmlcontent"])

                    with gr.Tab(label='P2P Network'):
                        with gr.Group() as p2p_panel:
                            p2p_active_checkbox = gr.Checkbox(label='Enable P2P network', value=ads.get_admin_default('p2p_active_checkbox'), info=shared.token.get_p2p_address())
                            p2p_remote_process = gr.Radio(label='Remote process', choices=['Disable', 'out', 'in'], value=ads.get_admin_default('p2p_remote_process'), interactive=ads.get_admin_default('p2p_active_checkbox'))
                            with gr.Group(visible=True if ads.get_admin_default('p2p_remote_process')=='out' else False) as p2p_out:
                                p2p_out_did_title = gr.Markdown(value="Remote node:", elem_classes=["p2p_title"])
                                with gr.Row():    
                                    p2p_out_did_input = gr.Textbox(max_lines=1, container=False, placeholder="Type did here.", min_width=60, elem_classes='p2p_input1')
                                    p2p_out_did_btn = gr.Button(value="Add", size="sm", min_width=30)
                                p2p_out_did_list = gr.Markdown(elem_classes=["htmlcontent"])
                            with gr.Group(visible=True if ads.get_admin_default('p2p_remote_process')=='in' else False) as p2p_in:
                                p2p_in_did_title = gr.Markdown(value="Accessible identity:", elem_classes=["p2p_title"])
                                with gr.Row():
                                    p2p_in_did_input = gr.Textbox(max_lines=1, container=False, placeholder="Type did here.", min_width=60, elem_classes='p2p_input1')
                                    p2p_in_did_btn = gr.Button(value="Add", size="sm", min_width=30)
                                p2p_in_did_list = gr.Markdown(elem_classes=["htmlcontent"])
                            with gr.Group() as p2p_ping:
                                p2p_ping_title = gr.Markdown(value="Send ping to:", elem_classes=["p2p_title"])
                                with gr.Row():
                                    p2p_ping_input = gr.Textbox(max_lines=1, container=False, placeholder="Type did:text here.", min_width=60, elem_classes='p2p_input1')
                                    p2p_ping_btn = gr.Button(value="Send", size="sm", min_width=30)
                                p2p_ping_result = gr.Markdown(elem_classes=["htmlcontent"])

                            p2p_out_did_btn.click(lambda x: x, inputs=p2p_out_did_input, outputs=p2p_out_did_list, queue=False, show_progress=False)
                            p2p_in_did_btn.click(lambda x: x, inputs=p2p_in_did_input, outputs=p2p_in_did_list, queue=False, show_progress=False)
                            p2p_out_did_list.change(lambda x,y: ads.set_admin_default_value("p2p_out_did_list", x, y), inputs=[p2p_out_did_list, state_topbar])
                            p2p_in_did_list.change(lambda x,y: ads.set_admin_default_value("p2p_in_did_list", x, y), inputs=[p2p_in_did_list, state_topbar])
                            
                            def toggle_p2p_remote_process(remote_process_status, state):
                                if remote_process_status!='Disable':
                                    p2p_task.init_p2p_task(worker, model_management, shared.token, minicpm)
                                ads.set_admin_default_value("p2p_remote_process", remote_process_status, state)
                                return [gr.update(visible=True if remote_process_status=='out' else False), gr.update(visible=True if remote_process_status=='in' else False)]

                            p2p_remote_process.change(toggle_p2p_remote_process, inputs=[p2p_remote_process, state_topbar], outputs=[p2p_out, p2p_in], queue=False, show_progress=False)
                            p2p_ping_btn.click(simpleai.ping_test, inputs=[p2p_ping_input, state_topbar], outputs=p2p_ping_result, queue=False, show_progress=False)

            with gr.Tab(label='Contact', elem_id="scrollable-box"):
                with gr.Row():
                    simpleai_contact = gr.HTML(visible=True, value=simpleai.self_contact, elem_classes=["identityIntroduce"], elem_id='identity_introduce')
                with gr.Row():
                    gr.Markdown(value=f'<b>系统版本</b>({version.get_simplesdxl_short_ver()})<br>OS: {shared.sysinfo["os_name"]}, {shared.sysinfo["cpu_arch"]}, {shared.sysinfo["cuda"]}, Torch{shared.sysinfo["torch_version"]}, XF{shared.sysinfo["xformers_version"]}<br>Ver: {version.branch} {version.simplesdxl_ver} / Comfyd {comfy_version.version}<br>PyHash: {shared.sysinfo["pyhash"]}, UIHash: {shared.sysinfo["uihash"]}')


        
                def sync_state_params(key, value, state):
                    state.update({key: value})
                    ads.set_user_default_value(key, value, state)

                def sync_backend_params(key, v, params, state):
                    params.update({key:v})
                    logger.debug(f'sync_backend_params: {key}:{v}')
                    if not key.startswith("hires_fix"):
                        ads.set_user_default_value(key, v, state) 
                    return params

                def toggle_minicpm(x, state):
                    MiniCPM.set_enable(x)
                    ads.set_admin_default_value('minicpm_checkbox', x, state) 
                    return gr.update(visible=not x), gr.update(visible=x), gr.update(visible=x), gr.update(visible= x), gr.update(visible=not x), gr.update(visible=x), gr.update(visible=x), gr.update(value='⚡ Execute Instruction' if x else 'Describe this Image into Prompt'), gr.update(interactive=x), gr.update(interactive=x)

                translation_methods.change(lambda x,y: ads.set_admin_default_value('translation_methods',x,y), inputs=[translation_methods, state_topbar])
                backfill_prompt.change(lambda x,y: ads.set_user_default_value("backfill_prompt",x,y), inputs=[backfill_prompt, state_topbar])
                disable_preview.change(lambda x,y: ads.set_user_default_value("disable_preview", x, y), inputs=[disable_preview, state_topbar])
                disable_intermediate_results.change(lambda x,y: ads.set_user_default_value("disable_intermediate_results", x, y), inputs=[disable_intermediate_results, state_topbar])
                disable_seed_increment.change(lambda x,y: ads.set_user_default_value("disable_seed_increment", x, y), inputs=[disable_seed_increment, state_topbar])
                save_final_enhanced_image_only.change(lambda x,y: ads.set_user_default_value("save_final_enhanced_image_only", x, y), inputs=[save_final_enhanced_image_only, state_topbar])
                generate_image_grid.change(lambda x,y: ads.set_user_default_value("generate_image_grid", x, y), inputs=[generate_image_grid, state_topbar])
                black_out_nsfw.change(lambda x,y: ads.set_user_default_value("black_out_nsfw", x, y), inputs=[black_out_nsfw, state_topbar])
                save_metadata_to_images.change(lambda x,y: ads.set_user_default_value("save_metadata_to_images", x, y), inputs=[save_metadata_to_images, state_topbar])
                metadata_scheme.change(lambda x,y: ads.set_user_default_value("metadata_scheme", x, y), inputs=[metadata_scheme, state_topbar])

                fast_comfyd_checkbox.change(simpleai.start_fast_comfyd, inputs=[fast_comfyd_checkbox, state_topbar])
                cache_clear_on_finish_checkbox.change(simpleai.set_cache_clear_on_finish, inputs=[cache_clear_on_finish_checkbox, state_topbar])
                minicpm_checkbox.change(toggle_minicpm, inputs=[minicpm_checkbox, state_topbar], outputs=[describe_apply_styles, describe_output_tags, describe_output_chinese, describe_output_artist, describe_methods, describe_prompt, vlm_describe_col, describe_btn, qwen_design_expand_btn, qwen_custom_expand_btn], queue=False, show_progress=False).then(None, _js="() => localizeWholePage()")
                minicpm_version.change(fn=lambda version, state: [minicpm.set_version(version), ads.set_admin_default_value('minicpm_version', version, state), gr.update(value=f'<div style="margin-bottom: 5px;">🤖 <b>VLM Model:</b> <span style="color: #2196F3;">{version}</span></div>')][-1], inputs=[minicpm_version, state_topbar], outputs=vlm_status_info)
                reserved_vram.change(lambda x,y: ads.set_admin_default_value('reserved_vram',x,y), inputs=[reserved_vram, state_topbar])
                cache_ram.change(lambda x,y: ads.set_admin_default_value('cache_ram',x,y), inputs=[cache_ram, state_topbar])
                advanced_logs.change(simpleai.change_advanced_logs, inputs=[advanced_logs, state_topbar])
                wavespeed_strength.change(lambda x,y: ads.set_admin_default_value('wavespeed_strength',x,y), inputs=[wavespeed_strength, state_topbar])
                admin_sync_button.click(topbar.admin_sync_to_guest, inputs=[state_topbar], outputs=admin_sync_button, queue=False, show_progress=False)

                admin_ctrls = [comfyd_active_checkbox, fast_comfyd_checkbox, cache_clear_on_finish_checkbox, reserved_vram, cache_ram, minicpm_checkbox, minicpm_version, advanced_logs, wavespeed_strength, translation_methods, p2p_active_checkbox, p2p_remote_process, p2p_in_did_list, p2p_out_did_list, no_welcome_checkbox, missing_model_filter_checkbox]
                user_app_ctrls = [backfill_prompt, image_tools_checkbox, disable_preview, disable_intermediate_results, disable_seed_increment, save_final_enhanced_image_only, style_preview_checkbox, generate_image_grid, black_out_nsfw, save_metadata_to_images, metadata_scheme, no_model_modal_checkbox]


            iclight_enable.change(lambda x: [gr.update(interactive=x, value='' if not x else comfy_task.iclight_source_names[0]), gr.update(value=flags.add_ratio('1024*1024') if not x else modules.config.default_aspect_ratio)], inputs=iclight_enable, outputs=[iclight_source_radio, aspect_ratios_selections[0]], queue=False, show_progress=False)
            layout_image_tab = [performance_selection, style_selections, image_number, freeu_enabled, refiner_model, refiner_switch] + lora_ctrls
            def toggle_image_tab(tab, styles):
                result = []
                if 'layer' in tab:
                    result += [gr.update(choices=flags.Performance.list()[:2]), gr.update(value=[s for s in styles if s!=fooocus_expansion and s!='Fooocus Sharp']), gr.update()]
                    result += [gr.update(value=False, interactive=False)]
                    result += [gr.update(interactive=False)] * 26
                elif 'uov' in tab:
                    result += [gr.update(choices=flags.Performance.list()), gr.update(), 1]
                    result += [gr.update(interactive=True)] * 27
                else:
                    result += [gr.update(choices=flags.Performance.list()), gr.update(), gr.update()]
                    result += [gr.update(interactive=True)] * 27
                return result
            
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            layer_tab.select(lambda: 'layer', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False).then(toggle_image_tab,inputs=[current_tab, style_selections], outputs=layout_image_tab, show_progress=False, queue=False)

            def toggle_image_input_panel(is_checked, is_tts_checked):
                result = [
                    gr.update(visible=is_checked),
                    gr.update(visible=is_checked),
                    gr.update(choices=flags.Performance.list()),
                    gr.update(),
                    gr.update(),
                ] + [gr.update(interactive=True)] * 27
                if is_checked:
                    result += [gr.update(visible=False), gr.update(value=False)]
                else:
                    result += [gr.update(), gr.update()]
                return result

            input_image_checkbox.change(
                toggle_image_input_panel,
                inputs=[input_image_checkbox, qwen_tts_checkbox],
                outputs=[image_input_panel, engine_class_display] + layout_image_tab + [tts_panel, qwen_tts_checkbox],
                queue=False,
                show_progress=False,
                _js=switch_js_two
            )

            def toggle_tts_panel(tts_checked, image_panel_checked):
                if tts_checked:
                    return gr.update(visible=True), gr.update(visible=False), gr.update(value=False)
                return gr.update(visible=False), gr.update(visible=image_panel_checked), gr.update()

            qwen_tts_checkbox.change(
                fn=toggle_tts_panel,
                inputs=[qwen_tts_checkbox, input_image_checkbox],
                outputs=[tts_panel, image_input_panel, input_image_checkbox],
                queue=False,
                show_progress=False,
                _js="(x,y) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);} return [x,y];}"
            )
            prompt_panel_checkbox.change(lambda x: [gr.update(visible=x, open=x if x else True), gr.update(visible=x)],
                                         inputs=prompt_panel_checkbox, outputs=[prompt_wildcards, prompt_history], queue=False, show_progress=False,
                                         _js=switch_js).then(
                                         lambda x,y: wildcards_array_show(y) if x else wildcards_array_hidden(),
                                         inputs=[prompt_panel_checkbox, state_topbar],
                                         outputs=wildcards_array, queue=False, show_progress=False)

            def toggle_comfyd_checked(x, state):
                ads.set_admin_default_value('comfyd_active_checkbox', x, state)
                if not args_manager.args.disable_backend and not args_manager.args.disable_comfyd:
                    comfyd.active(x)
                return

            image_tools_checkbox.change(lambda x,y: gr.update(visible=x or 'scene_frontend' in y) if "gallery_state" in y and y["gallery_state"] == 'finished_index' else gr.update(visible=False), inputs=[image_tools_checkbox,state_topbar], outputs=image_toolbox, queue=False, show_progress=False)
            comfyd_active_checkbox.change(toggle_comfyd_checked, inputs=[comfyd_active_checkbox, state_topbar], queue=False, show_progress=False)
            
            import enhanced.superprompter
            super_prompter.click(
                lambda x, y, z, c, i, i2, s, state_is_generating:
                    (logger.info('Using superprompter'), enhanced.superprompter.answer(input_text=enhanced.translator.convert(f'{y}{x}', z)))[1] if check_generating_state(state_is_generating) else
                    (logger.info('Using VLM'), minicpm.extended_prompt(x, y, [extract_scene_image(c), extract_scene_image(i), extract_scene_image(i2)], s, z))[1],
                inputs=[prompt, super_prompter_prompt, translation_methods, scene_canvas_image, scene_input_image1, scene_input_image2, state_topbar, state_is_generating],
                outputs=prompt,
                queue=False,
                show_progress=True
            )
            scene_params = [scene_theme, scene_canvas_image, scene_input_image1, scene_input_image2, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio, scene_image_number, scene_mask_color, scene_use_lora, scene_video, scene_audio]       
            scene_preset_save_ctrls = [scene_theme, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio, scene_image_number, scene_mask_color, scene_use_lora]

            language_ui.select(lambda x,y: sync_state_params('__lang', modules.config.language_radio_revert(x), y), inputs=[language_ui, state_topbar]).then(None, inputs=language_ui, _js="(x) => set_language_by_ui(x)")
            background_theme.select(lambda x,y: sync_state_params('__theme', x, y), inputs=[background_theme, state_topbar]).then(None, inputs=background_theme, _js="(x) => set_theme_by_ui(x)")

            gallery_index.change(gallery_util.images_list_update, inputs=[gallery_index, image_tools_checkbox, state_topbar], outputs=[gallery, progress_video, index_radio, image_toolbox, prompt_info_box, prompt_info_close_btn, prompt_info_container, progress_window, progress_gallery, identity_dialog, params_note_info, params_note_close_button, params_note_input_name, params_note_delete_button, params_note_regen_button, params_note_preset_button, params_note_box], show_progress=False)
            #gallery_index.select(gallery_util.select_index, inputs=[gallery_index, image_tools_checkbox, state_topbar], outputs=[gallery, progress_video, index_radio, image_toolbox, prompt_info_box, prompt_info_close_btn, prompt_info_container, progress_window, progress_gallery, identity_dialog, params_note_info, params_note_close_button, params_note_input_name, params_note_delete_button, params_note_regen_button, params_note_preset_button, params_note_box], show_progress=False)
            gallery.select(gallery_util.select_gallery, inputs=[gallery_index, state_topbar, backfill_prompt], outputs=[prompt_info_box, prompt_info_close_btn, prompt_info_container, prompt, negative_prompt, params_note_info, params_note_close_button, params_note_input_name, params_note_delete_button, params_note_regen_button, params_note_preset_button, params_note_box, state_topbar], show_progress=False)
            progress_gallery.select(gallery_util.select_gallery_progress, inputs=state_topbar, outputs=[prompt_info_box, prompt_info_close_btn, prompt_info_container, params_note_info, params_note_close_button, params_note_input_name, params_note_delete_button, params_note_regen_button, params_note_preset_button, params_note_box], show_progress=False)

        load_data_outputs = [progress_window, progress_gallery, progress_video, gallery, gallery_index, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + freeu_ctrls + lora_ctrls + [enhance_checkbox, enhance_enabled_1, enhance_enabled_2, enhance_enabled_3, enhance_uov_method, enhance_uov_strength]


        def inpaint_engine_state_change(inpaint_engine_version, state, *args):
            if inpaint_engine_version == 'empty':
                backend_engine = state.get('backend_engine', 'Z-image')
                if backend_engine == 'Z-image':
                    task_method = 'z_image_turbo_aio_cn'
                else:
                    task_method = 'SDXL' if 'task_method' not in state else state["task_method"]
                inpaint_engine_version = modules.flags.default_inpaint_engine_versions(task_method)

            result = []
            for inpaint_mode in args:
                if inpaint_mode != modules.flags.inpaint_option_detail:
                    result.append(gr.update(value=inpaint_engine_version))
                else:
                    result.append(gr.update())

            return result

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        
        def reset_aspect_ratios(aspect_ratios, use_resolution_override):
            if use_resolution_override:
                return [gr.update(visible=False)] * len(flags.aspect_ratios_templates)

            template = None
            if len(aspect_ratios.split(',')) > 1:
                template = aspect_ratios.split(',')[1]
                aspect_ratios = aspect_ratios.split(',')[0]
            else:
                template = 'SDXL'

            if template:
                results = []
                is_hit = False
                for aspect_ratio_name in flags.aspect_ratios_templates:
                    if template == aspect_ratio_name:
                        if aspect_ratios:
                            results.append(gr.update(value=aspect_ratios, visible=True))
                        else:
                            results.append(gr.update(visible=True))
                        is_hit = True
                    else:
                        results.append(gr.update(visible=False))
                if not is_hit:
                    results = [gr.update()] * len(flags.aspect_ratios_templates)
            else:
                results = [gr.update()] * len(flags.aspect_ratios_templates)
            return results


        aspect_ratios_selection.change(reset_aspect_ratios, inputs=[aspect_ratios_selection, use_resolution_override_checkbox], outputs=aspect_ratios_selections, queue=False, show_progress=False).then(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')

        def toggle_resolution_override(use_override, aspect_ratios):
            aspect_updates = reset_aspect_ratios(aspect_ratios, use_override)
            
            w_val = -1
            h_val = -1
            if use_override:
                try:
                    res = aspect_ratios.split(',')[0]
                    if '×' in res:
                        width_height = res.split('×')
                        w_val = int(width_height[0].strip())
                        h_val = int(width_height[1].split(' ')[0].strip())
                    elif '*' in res:
                        parts = res.split('*')
                        w_val = int(parts[0])
                        h_val = int(parts[1])
                except Exception as e:
                    pass
            
            return [gr.update(visible=use_override)] + aspect_updates + [gr.update(value=w_val), gr.update(value=h_val)]

        use_resolution_override_checkbox.change(
            toggle_resolution_override,
            inputs=[use_resolution_override_checkbox, aspect_ratios_selection],
            outputs=[resolution_override] + aspect_ratios_selections + [overwrite_width, overwrite_height],
            queue=False,
            show_progress=False
        )


        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        outpaint_selections.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state, outpaint_selections, state_topbar], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)
        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state, outpaint_selections, state_topbar], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        for mode, disable_initial_latent, engine, strength, respective_field in enhance_inpaint_update_ctrls:
            shared.gradio_root.load(enhance_inpaint_mode_change, inputs=[mode, inpaint_engine_state, state_topbar], outputs=[
                disable_initial_latent, engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image_full, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image_full]
        ctrls += [outpaint_selections, inpaint_input_image_backend, inpaint_additional_prompt, inpaint_mask_image_backend]
        ctrls += [layer_method, layer_input_image_full, iclight_enable, iclight_source_radio]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls
        ctrls += [params_backend]
        ctrls += [save_final_enhanced_image_only if not args_manager.args.disable_image_log else None]
        ctrls += [save_metadata_to_images if not args_manager.args.disable_metadata else None]
        ctrls += [metadata_scheme if not args_manager.args.disable_metadata else None]
        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image_full, enhance_checkbox, enhance_uov_method, enhance_uov_strength, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls
        # ctrls += [random_aspect_ratio_checkbox]

        def parse_meta(raw_prompt_txt, state_params, scene_input_image1, state_is_generating):
            if state_is_generating:
                 return [gr.update()]*5
            if len(raw_prompt_txt)>=1 and (raw_prompt_txt[-1]=='[' or raw_prompt_txt[-1]=='_'):
                return [gr.update()] * 4 + [True]
            super_prompter_result = gr.update(interactive=len(raw_prompt_txt)>0)
            if 'scene_frontend' in state_params and len(raw_prompt_txt)==0 and scene_input_image1 is not None:
                is_canvas_image = 'scene_canvas_image' not in state_params["scene_frontend"].get('disvisible', [])
                if not is_canvas_image:
                    return [gr.update(), super_prompter_result, gr.update(visible=False), gr.update(visible=True), gr.update()]
            
            return [gr.update(), super_prompter_result, gr.update(visible=True), gr.update(visible=False), gr.update()]

        prompt.change(parse_meta, inputs=[prompt, state_topbar, scene_input_image1, state_is_generating], outputs=[prompt, super_prompter, generate_button, load_parameter_button, prompt_panel_checkbox], queue=False, show_progress=False)      

        def trigger_metadata_import(file, state_is_generating, state_params):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                logger.info('Could not find metadata in the image!')
            return toolbox.reset_params_by_image_meta(parameters, state_params, state_is_generating, inpaint_mode)

        def update_prompt_history(task, existing_history):
            MAX_HISTORY = 5
            if not task or not hasattr(task, 'final_prompts'):
                return existing_history[-MAX_HISTORY:]

            new_unique_prompts = []
            for p in task.final_prompts:
                if p and p.strip() and p not in new_unique_prompts:
                    new_unique_prompts.append(p)
            for p in new_unique_prompts:
                while p in existing_history:
                    existing_history.remove(p)
            updated_history = existing_history + new_unique_prompts
            return updated_history[-MAX_HISTORY:]

        image_input_panel_ctrls = [engine_class_display, uov_method, layer_method, layer_input_image, enhance_checkbox, enhance_input_image]
        reset_preset_layout = [params_backend, advanced_checkbox, performance_selection, scheduler_name, sampler_name, input_image_checkbox, prompt_panel_checkbox, enhance_checkbox, base_model, refiner_model, overwrite_step, guidance_scale, negative_prompt, preset_instruction, identity_dialog] + image_input_panel_ctrls + lora_ctrls
        reset_preset_func = [output_format, inpaint_advanced_masking_checkbox, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, backfill_prompt, translation_methods, input_image_checkbox, quick_enhance]
        scene_frontend_ctrls = [prompt_internal_panel, random_button, super_prompter, disable_intermediate_results, image_tools_checkbox, scene_panel, scene_theme] + scene_params[1:] + [generate_button, load_parameter_button]

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating, state_topbar], outputs=reset_preset_layout + reset_preset_func + scene_frontend_ctrls + load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        model_check = [prompt, negative_prompt, base_model, refiner_model] + lora_ctrls

        def process_image_for_html(img):
            if img is None: return None
            if isinstance(img, str): return f"/file={img}"
            if isinstance(img, np.ndarray):
                # Resize image if it's too large for preview
                h, w = img.shape[:2]
                max_side = 2048
                if h > max_side or w > max_side:
                    scale = max_side / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    pil_img = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    pil_img = Image.fromarray(img)
                
                if pil_img.mode == 'RGBA':
                    pil_img = Image.alpha_composite(Image.new("RGBA", pil_img.size, (255, 255, 255, 255)), pil_img).convert("RGB")
                
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            return None

        def cache_input_image_func(state_params, tab, uov, inpaint, layer, enhance, scene1, scene_canvas):
            img = None
            is_scene = isinstance(state_params, dict) and ("scene_frontend" in state_params)

            if is_scene:
                if scene_canvas is not None:
                    if isinstance(scene_canvas, dict):
                        img = scene_canvas.get('image')
                    else:
                        img = scene_canvas
                if img is None and scene1 is not None:
                    img = scene1
                return process_image_for_html(img)

            if tab == 'uov' or tab == 'uov_tab':
                img = uov
            elif tab == 'inpaint' or tab == 'inpaint_tab':
                if isinstance(inpaint, dict):
                    img = inpaint.get('image')
                else:
                    img = inpaint
            elif tab == 'layer' or tab == 'layer_tab':
                img = layer
            elif tab == 'enhance' or tab == 'enhance_tab':
                img = enhance
            else:
                img = None

            return process_image_for_html(img)

        def toggle_comparison(is_comp, input_img, gallery_output, final_gallery):
            if is_comp:
                 # Switch back to Gallery
                 return False, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            if not input_img:
                return False, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            output_img = None
            
            # Try to find output image from progress gallery first, then final gallery
            for output in [gallery_output, final_gallery]:
                if output and len(output) > 0:
                    first_item = output[0]
                    if isinstance(first_item, (list, tuple)):
                        img_data = first_item[0]
                        if isinstance(img_data, dict):
                             output_img = img_data.get('name') or img_data.get('data')
                        else:
                             output_img = img_data
                    elif isinstance(first_item, dict):
                        output_img = first_item.get('name') or first_item.get('data')
                    else:
                        output_img = first_item
                    
                    if output_img:
                        break

            if not output_img:
                return False, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            output_img_url = f"/file={output_img}" if isinstance(output_img, str) and not output_img.startswith("data:") else output_img


            input_img_url = input_img
            if isinstance(input_img, str) and not input_img.startswith("data:") and not input_img.startswith("/file="):
                 input_img_url = f"/file={input_img}"

            import time
            unique_id = f"comp_{int(time.time() * 1000)}"
            html = f"""
            <div id="{unique_id}" class="comparison-wrapper"
                onclick="
                    var isEntering = !this.classList.contains('fullscreen-mode');
                    if (isEntering) {{
                        try {{
                            this.__lf_comp_parent = this.parentNode;
                            this.__lf_comp_placeholder = document.createElement('span');
                            this.__lf_comp_placeholder.style.display = 'none';
                            this.__lf_comp_parent.insertBefore(this.__lf_comp_placeholder, this.nextSibling);
                            document.body.appendChild(this);
                            document.body.style.overflow = 'hidden';
                        }} catch (e) {{}}
                    }} else {{
                        try {{
                            document.body.style.overflow = '';
                            if (this.__lf_comp_parent && this.__lf_comp_placeholder) {{
                                this.__lf_comp_parent.insertBefore(this, this.__lf_comp_placeholder);
                                this.__lf_comp_placeholder.remove();
                            }}
                        }} catch (e) {{}}
                        this.__lf_comp_parent = null;
                        this.__lf_comp_placeholder = null;
                    }}
                    this.classList.toggle('fullscreen-mode');
                    var outer = this.querySelector('.outer-img');
                    var inner = this.querySelector('.inner-img');
                    if(outer && inner) {{
                        inner.style.width = outer.offsetWidth + 'px';
                        inner.style.height = outer.offsetHeight + 'px';
                    }}
                "
                onmousemove="
                    var outer = this.querySelector('.outer-img');
                    var overlay = this.querySelector('.overlay');
                    if(!outer || !overlay) return;
                    var rect = outer.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    x = Math.max(0, Math.min(rect.width, x));
                    overlay.style.width = x + 'px';
                ">
                <div class="comparison-content">
                    <img class="comp-img outer-img" src="{output_img_url}" draggable="false" 
                        onload="
                            var wrapper = this.closest('.comparison-wrapper');
                            var inner = wrapper.querySelector('.inner-img');
                            if(inner) {{
                                inner.style.width = this.offsetWidth + 'px';
                                inner.style.height = this.offsetHeight + 'px';
                            }}
                        "
                    />
                    <div class="overlay">
                        <img class="comp-img inner-img" src="{input_img_url}" draggable="false" />
                    </div>
                    <div class="label" style="left: 10px;">Input</div>
                    <div class="label" style="right: 10px;">Output</div>
                </div>
                <!-- Inline script to handle resize and initial sync more robustly -->
                <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" 
                     style="display:none" 
                     onload="
                        var wrapper = document.getElementById('{unique_id}');
                        if(wrapper) {{
                             var outer = wrapper.querySelector('.outer-img');
                             var inner = wrapper.querySelector('.inner-img');
                             function sync() {{
                                 if(outer && inner && outer.offsetWidth > 0) {{
                                     inner.style.width = outer.offsetWidth + 'px';
                                     inner.style.height = outer.offsetHeight + 'px';
                                 }}
                             }}
                             sync();
                             if(window.ResizeObserver) {{
                                 const ro = new ResizeObserver(sync);
                                 ro.observe(outer);
                             }} else {{
                                 setInterval(sync, 100); 
                                 window.addEventListener('resize', sync);
                             }}
                        }}
                     "
                />
            </div>
            """
            return True, gr.update(value=html, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        def check_comparison_visibility(input_img, gallery_output, state_topbar):
            engine_type = state_topbar.get('engine_type')
            if not engine_type:
                engine_type = state_topbar.get('default_engine', {}).get('engine_type')

            if engine_type == 'video':
                return gr.update(visible=False, size='sm')
            if input_img and gallery_output and len(gallery_output) > 0:
                 return gr.update(visible=True, size='sm')
            return gr.update(visible=False, size='sm')

        compare_btn.click(toggle_comparison, inputs=[comparison_state, cached_input_image, progress_gallery, gallery], outputs=[comparison_state, comparison_box, progress_gallery, gallery, progress_window, progress_video], show_progress=False)
        protections = [random_button, super_prompter, background_theme, image_tools_checkbox] + nav_bars
        from extras.media_normalize import stash_scene_media_before_generation as _stash_scene_media_before_generation
        from extras.media_normalize import stash_scene_media_preview as _stash_scene_media_preview
        from extras.media_normalize import restore_scene_media_after_generation as _restore_scene_media_after_generation

        generate_button.click(_stash_scene_media_before_generation, inputs=[scene_video, scene_audio, scene_original_video_path, state_topbar], outputs=[scene_video_backup, scene_audio_backup, scene_original_video_backup, scene_video, scene_audio, scene_original_video_path, scene_video_placeholder, scene_audio_placeholder, generate_button, skip_button, stop_button, random_aspect_ratio_state], queue=False, show_progress=False) \
            .then(compose_full_sketch, inputs=[inpaint_input_image, inpaint_input_image_full], outputs=[inpaint_input_image_backend], queue=False, show_progress=False) \
            .then(compose_full_mask, inputs=[inpaint_mask_image, inpaint_input_image_full], outputs=[inpaint_mask_image_backend], queue=False, show_progress=False) \
            .then(compose_full_sketch, inputs=[scene_canvas_image, scene_canvas_image_full], outputs=[scene_canvas_image_backend], queue=False, show_progress=False) \
            .then(cache_input_image_func, inputs=[state_topbar, current_tab, uov_input_image_full, inpaint_input_image_backend, layer_input_image_full, enhance_input_image_full, scene_input_image1_full, scene_canvas_image_backend], outputs=[cached_input_image]) \
            .then(topbar.process_before_generation, inputs=[state_topbar, seed_random, image_seed, params_backend, scene_theme, scene_canvas_image_backend, scene_input_image1_full, scene_input_image2_full, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio, scene_image_number, scene_video_backup, scene_audio_backup, scene_original_video_backup, active_video_source, sam3_input_video, sam3_original_video_path, sam3_mask_video], outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box, image_seed] + protections + [preset_store, identity_dialog], show_progress=False) \
            .then(topbar.wait_for_minicpm_completion, outputs=[], show_progress=False) \
            .then(topbar.avoid_empty_prompt_for_scene, inputs=[prompt, state_topbar, scene_canvas_image_backend, scene_input_image1_full, scene_theme, scene_additional_prompt, scene_additional_prompt_2], outputs=prompt, show_progress=True) \
            .then(lambda state_topbar_value, use_loras, model1, model2, model3, model4: [ \
                 "None" if "scene_frontend" in state_topbar_value and not use_loras else model1, "None" if "scene_frontend" in state_topbar_value and not use_loras else model2, "None" if "scene_frontend" in state_topbar_value and not use_loras else model3, "None" if "scene_frontend" in state_topbar_value and not use_loras else model4], \
            inputs=[state_topbar, scene_use_lora, scene_lora_model, scene_lora_model_2, scene_lora_model_3, scene_lora_model_4], \
            outputs=[scene_lora_model, scene_lora_model_2, scene_lora_model_3, scene_lora_model_4]) \
            .then(select_random_aspect_ratio, inputs=[random_aspect_ratio_checkbox, random_aspect_ratio_state], outputs=[overwrite_width, overwrite_height, aspect_ratios_selection, random_aspect_ratio_state]) \
            .then(fn=get_task_with_resolution_multiplier, inputs=ctrls + [resolution_multiplier, resolution_quantize_step], outputs=currentTask) \
            .then(fn=generate_clicked, inputs=[currentTask, state_topbar], outputs=[progress_html, progress_window, progress_gallery, progress_video, gallery, comparison_state, comparison_box, compare_btn, stop_button, skip_button]) \
            .then(topbar.process_after_generation, inputs=state_topbar, outputs=[generate_button, stop_button, skip_button, state_is_generating, gallery_index, index_radio] + protections + [gallery_index_stat, history_link], show_progress=False) \
            .then(check_comparison_visibility, inputs=[cached_input_image, progress_gallery, state_topbar], outputs=[compare_btn]) \
            .then(_restore_scene_media_after_generation, inputs=[state_topbar, scene_video_backup, scene_audio_backup, scene_original_video_backup], outputs=[scene_video, scene_audio, scene_original_video_path, scene_video_placeholder, scene_audio_placeholder], queue=False, show_progress=False) \
            .then(lambda x: None, inputs=gallery_index_stat, queue=False, show_progress=False, _js='(x)=>{refresh_finished_images_catalog_label(x);}') \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed') \
            .then(fn=update_prompt_history,inputs=[currentTask, state_prompt_history],outputs=state_prompt_history) \
            .then(lambda h: gr.Dataset.update(samples=[[v] for v in h]),inputs=state_prompt_history,outputs=history_prompts)

        debug_true_state = gr.State(value=True)
        ctrls_preview = [debug_true_state if c == debugging_cn_preprocessor else c for c in ctrls]

        preview_preprocessing.click(_stash_scene_media_preview, inputs=[scene_video, scene_audio, scene_original_video_path], outputs=[scene_video_backup, scene_audio_backup, scene_original_video_backup, scene_video, scene_audio, scene_original_video_path, scene_video_placeholder, scene_audio_placeholder], queue=False, show_progress=False) \
            .then(lambda: (False, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=None, visible=True), gr.update(visible=False, size='sm')), outputs=[comparison_state, comparison_box, progress_window, gallery, progress_gallery, compare_btn]) \
            .then(compose_full_sketch, inputs=[inpaint_input_image, inpaint_input_image_full], outputs=[inpaint_input_image_backend], queue=False, show_progress=False) \
            .then(compose_full_mask, inputs=[inpaint_mask_image, inpaint_input_image_full], outputs=[inpaint_mask_image_backend], queue=False, show_progress=False) \
            .then(compose_full_sketch, inputs=[scene_canvas_image, scene_canvas_image_full], outputs=[scene_canvas_image_backend], queue=False, show_progress=False) \
            .then(topbar.process_before_generation, inputs=[state_topbar, seed_random, image_seed, params_backend, scene_theme, scene_canvas_image_backend, scene_input_image1_full, scene_input_image2_full, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio, scene_image_number, scene_video_backup, scene_audio_backup, scene_original_video_backup, active_video_source, sam3_input_video, sam3_original_video_path, sam3_mask_video], outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box, image_seed] + protections + [preset_store, identity_dialog], show_progress=False) \
            .then(fn=get_task_with_resolution_multiplier, inputs=ctrls_preview + [resolution_multiplier, resolution_quantize_step], outputs=currentTask) \
            .then(fn=generate_clicked, inputs=[currentTask, state_topbar], outputs=[progress_html, progress_window, progress_gallery, progress_video, gallery, comparison_state, comparison_box, compare_btn, stop_button, skip_button]) \
            .then(topbar.process_after_generation, inputs=state_topbar, outputs=[generate_button, stop_button, skip_button, state_is_generating, gallery_index, index_radio] + protections + [gallery_index_stat, history_link], show_progress=False) \
            .then(check_comparison_visibility, inputs=[cached_input_image, progress_gallery, state_topbar], outputs=[compare_btn]) \
            .then(_restore_scene_media_after_generation, inputs=[state_topbar, scene_video_backup, scene_audio_backup, scene_original_video_backup], outputs=[scene_video, scene_audio, scene_original_video_path, scene_video_placeholder, scene_audio_placeholder], queue=False, show_progress=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles, output_tags, output_chinese, output_artist, describe_prompt=""):
            if img is None and not MiniCPM.get_enable():
                logger.info("Image is None in trigger_describe and VLM is not enabled, skipping image description")
                return gr.update(), gr.update()

            describe_images = []
            styles = set()

            if img is not None and flags.describe_type_photo in modes and not MiniCPM.get_enable():
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_images.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if img is not None and flags.describe_type_anime in modes and (not MiniCPM.get_enable() or (MiniCPM.get_enable() and output_tags)):
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_images.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])
            
            if img is not None and (flags.describe_type_artist in modes and (not MiniCPM.get_enable()) or ((MiniCPM.get_enable() and output_artist))):
                artist_result = get_artist_tags_string(img, None)
                describe_images.append(artist_result)

            if MiniCPM.get_enable() and (img is None or (not output_tags and not output_artist and len(describe_images) == 0)):
                describe_images.append(minicpm.interrogate(img, output_chinese, additional_prompt=describe_prompt))
                styles.update([])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_images) == 0:
                describe_image = gr.update()
            else:
                describe_image = ', '.join(describe_images)

                if MiniCPM.get_enable() and (output_tags or output_artist) and output_chinese:
                    describe_image = minicpm.translate_cn(describe_image)

            return describe_image, styles

        def describe_with_generating_check(state_is_generating, modes, img, apply_styles, output_tags, output_chinese, output_artist, describe_prompt=""):
            is_worker_processing = modules.async_worker.worker_processing is not None
            has_pending_tasks = modules.async_worker.pending_tasks > 0

            if check_generating_state(state_is_generating, has_pending_tasks, is_worker_processing):
                logger.info("Generation is in progress or pending, skipping image description")
                return gr.update(), gr.update()

            return trigger_describe(modes, img, apply_styles, output_tags, output_chinese, output_artist, describe_prompt)
        describe_btn.click(describe_with_generating_check,
                           inputs=[state_is_generating, describe_methods, describe_input_image, describe_apply_styles,
                                   describe_output_tags, describe_output_chinese, describe_output_artist, describe_prompt],
                           outputs=[prompt, style_selections],
                           show_progress=True,
                           queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        def unload_models_clicked(state_is_generating):
            is_worker_processing = modules.async_worker.worker_processing is not None
            has_pending_tasks = modules.async_worker.pending_tasks > 0

            if check_generating_state(state_is_generating, has_pending_tasks, is_worker_processing):
                logger.info("Generation is in progress or pending, skipping model unload")
                return

            minicpm.free_model()

            try:
                import extras.interrogate
                extras.interrogate.free_model()
            except Exception:
                pass

            try:
                import extras.wd14tagger
                extras.wd14tagger.free_model()
            except Exception:
                pass

            try:
                from enhanced.sam3_video_mask import unload_sam3_video_predictor
                unload_sam3_video_predictor()
            except Exception:
                pass

            try:
                from enhanced import webui_qwen_tts
                webui_qwen_tts.unload_qwen_tts_models()
            except Exception:
                pass

            model_management.unload_all_models()
            model_management.soft_empty_cache()
            logger.info("Models unloaded manually.")
            return

        unload_btn.click(unload_models_clicked, inputs=[state_is_generating], show_progress=True)

        def trigger_auto_describe_for_scene(state, canvas_image, img, scene_theme, additional_prompt, additional_prompt_2, state_is_generating):
            is_worker_processing = worker.worker_processing is not None
            has_pending_tasks = worker.pending_tasks > 0
            is_generating = state_is_generating or is_worker_processing or has_pending_tasks

            if is_generating:
                logger.info(f"Generation is in progress or pending, skipping image description")
                return gr.update(), gr.update(), gr.update()

            is_canvas_image = 'scene_canvas_image' not in state["scene_frontend"].get('disvisible', [])
            ready_to_gen = True 
            canvas_img = extract_scene_image(canvas_image) if is_canvas_image else None
            input_img = extract_scene_image(img)
            use_img = canvas_img if canvas_img is not None else input_img
            if use_img is None and is_canvas_image:
                ready_to_gen = False
            describe_prompt, img_is_ok = describe_prompt_for_scene(state, use_img, scene_theme, f'{additional_prompt}{additional_prompt_2}')
            styles = set()
            styles.update([])
            return describe_prompt if describe_prompt else gr.update(), list(styles), gr.update(interactive=ready_to_gen and img_is_ok)

        def trigger_auto_aspect_ratio_for_scene_from_canvas_image(state, canvas_image, input_image1, scene_theme, video=None, audio=None):
            img = None
            if isinstance(canvas_image, dict):
                img = canvas_image.get('image', None)
            elif isinstance(canvas_image, np.ndarray):
                img = canvas_image
            results = [trigger_auto_aspect_ratio_for_scene(state, img, scene_theme)]
            need_canvas_image = 'scene_canvas_image' not in state["scene_frontend"].get('disvisible', [])
            need_input_image1 = 'scene_input_image1' not in state["scene_frontend"].get('disvisible', [])
            need_input_image2 = 'scene_input_image2' not in state["scene_frontend"].get('disvisible', [])

            video_visible = 'scene_video' not in state["scene_frontend"].get('disvisible', [])
            audio_visible = 'scene_audio' not in state["scene_frontend"].get('disvisible', [])

            if video_visible and video is not None:
                 results.append(gr.update(interactive=True, visible=True))
                 return results
            if audio_visible and audio is not None:
                 results.append(gr.update(interactive=True, visible=True))
                 return results

            if need_canvas_image and canvas_image is not None:
                if need_input_image2 or (not need_input_image1 or (need_input_image1 and input_image1 is not None)):
                    results.append(gr.update(interactive=True, visible=True))
                else:
                    results.append(gr.update(interactive=False, visible=True))
            else:
                results.append(gr.update(interactive=False, visible=True))
            return results

        def trigger_auto_aspect_ratio_for_scene_from_input_image(state, input_image1, scene_theme):
            is_canvas_image = 'scene_canvas_image' not in state["scene_frontend"].get('disvisible', [])
            if is_canvas_image:
                return gr.update()
            return trigger_auto_aspect_ratio_for_scene(state, input_image1, scene_theme)

        def trigger_auto_aspect_ratio_for_scene(state, img, scene_theme):
            if img is None:
                return gr.update()
            img = resize_image(img, max_side=1280, resize_mode=4)
            aspect_ratios = modules.flags.get_value_by_scene_theme(state, scene_theme, 'aspect_ratio', [])
            aspect_ratio_select_mode = state['scene_frontend'].get('aspect_ratio_select_mode', '')
            if not aspect_ratio_select_mode:
                return gr.update()
            aspect_ratios_new, aspect_ratio = get_auto_candidate(img, aspect_ratios, aspect_ratio_select_mode)
            if aspect_ratio_select_mode:
                aspect_ratios = aspect_ratios_new
                if 'auto_match' in aspect_ratio_select_mode:
                    aspect_ratios = [aspect_ratio]
            aspect_ratios = modules.flags.scene_aspect_ratios_mapping_list(aspect_ratios)
            aspect_ratio = modules.flags.scene_aspect_ratios_mapping(aspect_ratio)
            return gr.update(choices=aspect_ratios, value=aspect_ratio)
        
        def scene_input_image1_clear(state, input_image1, video=None, audio=None):
            if input_image1 is None and 'scene_frontend' in state:
                scene_input_image1_visible = 'scene_input_image1' not in state["scene_frontend"].get('disvisible', [])
                scene_input_image2_visible = 'scene_input_image2' not in state["scene_frontend"].get('disvisible', [])
                need_canvas_image = 'scene_canvas_image' not in state["scene_frontend"].get('disvisible', [])
                should_disable_generate = scene_input_image1_visible and not (scene_input_image2_visible and need_canvas_image)

                video_visible = 'scene_video' not in state["scene_frontend"].get('disvisible', [])
                audio_visible = 'scene_audio' not in state["scene_frontend"].get('disvisible', [])
                if (video_visible and video is not None) or (audio_visible and audio is not None):
                     should_disable_generate = False

                if should_disable_generate:
                    return '', gr.update(interactive=False, visible=True), gr.update(visible=False)
            return [gr.update()]*3

        def scene_canvas_image_clear(state, canvas_image, input_image1):
            if canvas_image is None:
                print(f'scene_canvas_image_clear')
                need_canvas_image = 'scene_canvas_image' not in state["scene_frontend"].get('disvisible', [])
                need_input_image1 = 'scene_input_image1' not in state["scene_frontend"].get('disvisible', [])
                if need_canvas_image and canvas_image is not None and (not need_input_image1 or (need_input_image1 and input_image1 is not None)):
                    return gr.update(interactive=True, visible=True)
                else:
                    return gr.update(interactive=False, visible=True)
            else:
                return gr.update()
        def update_describe_output_tags(engine_class_display):
            if engine_class_display in ['SDXL', 'SD15', 'Illustrious']:
                return gr.update(value=True)
            return gr.update(value=False)

        def update_scene_model_dropdown_visibility(state):
            if not isinstance(state, dict):
                return gr.update(), gr.update()
            scenes = state.get("scene_frontend", {})
            if not isinstance(scenes, dict):
                return gr.update(), gr.update()
            disvisible = scenes.get("disvisible", [])
            if not isinstance(disvisible, list):
                disvisible = []
            return (
                gr.update(visible='scene_base_model' not in disvisible),
                gr.update(visible='scene_refiner_model' not in disvisible),
            )

        def sync_scene_model_selections(state, base_model_value, refiner_model_value, *lora_ctrl_values):
            if not isinstance(state, dict):
                state = {}

            base_choices = modules.config.model_filenames
            refiner_choices = ["None"] + base_choices
            lora_choices = ["None"] + modules.config.lora_filenames

            scenes = state.get("scene_frontend", {})
            disvisible = []
            if isinstance(scenes, dict):
                disvisible = scenes.get("disvisible", [])
            if not isinstance(disvisible, list):
                disvisible = []

            base_visible = 'scene_base_model' not in disvisible
            refiner_visible = 'scene_refiner_model' not in disvisible

            if base_model_value not in base_choices and base_choices:
                base_model_value = base_choices[0]
            if refiner_model_value not in refiner_choices:
                refiner_model_value = "None"

            parsed_loras = []
            for i in range(min(4, len(lora_ctrl_values) // 3)):
                enabled = lora_ctrl_values[i * 3]
                filename = lora_ctrl_values[i * 3 + 1]
                weight = lora_ctrl_values[i * 3 + 2]
                enabled = bool(enabled)
                if not isinstance(filename, str) or filename not in lora_choices:
                    filename = "None"
                if not isinstance(weight, (int, float)):
                    weight = 1.0
                parsed_loras.append((filename, weight))
            while len(parsed_loras) < 4:
                parsed_loras.append(("None", 1.0))

            return [
                gr.update(choices=base_choices, value=base_model_value, visible=base_visible),
                gr.update(choices=refiner_choices, value=refiner_model_value, visible=refiner_visible),
                gr.update(value=True),
                gr.update(visible=True),
                gr.update(choices=lora_choices, value=parsed_loras[0][0]),
                gr.update(value=parsed_loras[0][1]),
                gr.update(choices=lora_choices, value=parsed_loras[1][0]),
                gr.update(value=parsed_loras[1][1]),
                gr.update(choices=lora_choices, value=parsed_loras[2][0]),
                gr.update(value=parsed_loras[2][1]),
                gr.update(choices=lora_choices, value=parsed_loras[3][0]),
                gr.update(value=parsed_loras[3][1]),
            ]

        scene_canvas_image.upload(stash_preview_sketch, inputs=[scene_canvas_image], outputs=[scene_canvas_image, scene_canvas_image_full], show_progress=False, queue=False) \
                        .then(trigger_auto_aspect_ratio_for_scene_from_canvas_image, inputs=[state_topbar, scene_canvas_image, scene_input_image1, scene_theme, scene_video, scene_audio], outputs=[scene_aspect_ratio, generate_button], show_progress=False, queue=False) \
                        .then(lambda: None, _js='()=>{refresh_scene_localization();}')
        scene_canvas_image.clear(lambda: (None, None), outputs=[scene_canvas_image_full, scene_canvas_image_backend], show_progress=False, queue=False)
        #scene_canvas_image.change(scene_canvas_image_clear, inputs=[state_topbar, scene_canvas_image, scene_input_image1], outputs=[generate_button], show_progress=False, queue=False)
        scene_input_image1.upload(stash_preview_image, inputs=[scene_input_image1], outputs=[scene_input_image1, scene_input_image1_full], show_progress=False, queue=False) \
                        .then(trigger_auto_describe_for_scene, inputs=[state_topbar, scene_canvas_image, scene_input_image1_full, scene_theme, scene_additional_prompt, scene_additional_prompt_2, state_is_generating], outputs=[prompt, style_selections, generate_button], show_progress=True, queue=True) \
                        .then(trigger_auto_aspect_ratio_for_scene_from_input_image, inputs=[state_topbar, scene_input_image1_full, scene_theme],
                                outputs=scene_aspect_ratio, show_progress=False, queue=False) \
                        .then(lambda: None, _js='()=>{refresh_scene_localization();}')
        #scene_input_image1.clear(lambda: ['', gr.update(interactive=False)], outputs=[prompt, generate_button], show_progress=False, queue=False)
        scene_input_image1.change(scene_input_image1_clear, inputs=[state_topbar, scene_input_image1, scene_video, scene_audio], outputs=[prompt, generate_button, load_parameter_button], show_progress=False, queue=False) 
        scene_input_image1.change(lambda img, full: None if img is None else full, inputs=[scene_input_image1, scene_input_image1_full], outputs=[scene_input_image1_full], show_progress=False, queue=False)
        scene_input_image2.upload(stash_preview_image, inputs=[scene_input_image2], outputs=[scene_input_image2, scene_input_image2_full], show_progress=False, queue=False)
        scene_input_image2.change(lambda img, full: None if img is None else full, inputs=[scene_input_image2, scene_input_image2_full], outputs=[scene_input_image2_full], show_progress=False, queue=False)
        load_parameter_button.click(trigger_auto_describe_for_scene, inputs=[state_topbar, scene_canvas_image, scene_input_image1_full, scene_theme, scene_additional_prompt, scene_additional_prompt_2, state_is_generating], outputs=[prompt, style_selections, generate_button], show_progress=True, queue=False) \
                        .then(trigger_auto_aspect_ratio_for_scene, inputs=[state_topbar, scene_input_image1_full, scene_theme],
                                outputs=scene_aspect_ratio, show_progress=False, queue=False) \
                        .then(lambda: None, _js='()=>{refresh_scene_localization();}')

        scene_theme.select(switch_scene_theme_select, inputs=state_topbar, queue=False, show_progress=False).then(
            update_scene_model_dropdown_visibility,
            inputs=[state_topbar],
            outputs=[scene_base_model, scene_refiner_model],
            queue=False,
            show_progress=False,
        )
        scene_theme.change(switch_scene_theme, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_theme], outputs=scene_params[1:], queue=False, show_progress=False) \
                   .then(update_scene_model_dropdown_visibility, inputs=[state_topbar], outputs=[scene_base_model, scene_refiner_model], queue=False, show_progress=False) \
                   .then(switch_scene_theme_ready_to_gen, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_theme, scene_video, scene_audio], outputs=[prompt, generate_button], queue=False, show_progress=True) \
                   .then(check_camera_control_visibility, inputs=[scene_theme, state_topbar], outputs=[camera_control_accordion, anglelight_control_accordion, style_transfer_accordion, sam3_video_mask_accordion], queue=False, show_progress=False)

        scene_aspect_ratio.change(lambda: [gr.update(value=-1), gr.update(value=-1)], outputs=[overwrite_width, overwrite_height], queue=False, show_progress=False)
        scene_aspect_ratio.select(lambda: [gr.update(value=-1), gr.update(value=-1)], outputs=[overwrite_width, overwrite_height], queue=False, show_progress=False)

        scene_video.upload(switch_scene_theme_ready_to_gen, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_theme, scene_video, scene_audio], outputs=[prompt, generate_button], queue=False, show_progress=False)
        scene_video.clear(switch_scene_theme_ready_to_gen, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_theme, scene_video, scene_audio], outputs=[prompt, generate_button], queue=False, show_progress=False)
        scene_audio.upload(switch_scene_theme_ready_to_gen, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_theme, scene_video, scene_audio], outputs=[prompt, generate_button], queue=False, show_progress=False)
        scene_audio.clear(switch_scene_theme_ready_to_gen, inputs=[state_topbar, image_number, scene_canvas_image, scene_input_image1, scene_additional_prompt, scene_additional_prompt_2, scene_theme, scene_video, scene_audio], outputs=[prompt, generate_button], queue=False, show_progress=False)

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles, output_tags, output_chinese, output_artist, state_is_generating=True):

                if img is None:
                    logger.info("Image is None, skipping image description")
                    return gr.update(), gr.update()

                if isinstance(img, dict):
                    img = img['image']
                return trigger_describe(mode, img, apply_styles, output_tags, output_chinese, output_artist)

            uov_input_image.upload(lambda: None, outputs=[], show_progress=False, queue=False) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            uov_input_image.change(lambda img: gr.update(visible=img is not None), inputs=uov_input_image, outputs=describe_uov_button, show_progress=False, queue=False)
            uov_input_image.change(lambda img, full: None if img is None else full, inputs=[uov_input_image, uov_input_image_full], outputs=[uov_input_image_full], show_progress=False, queue=False)

            describe_uov_button.click(trigger_auto_describe, inputs=[describe_methods, uov_input_image_full, prompt, describe_apply_styles, describe_output_tags, describe_output_chinese, describe_output_artist], outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            inpaint_input_image.upload(lambda: None, outputs=[], show_progress=False, queue=False) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(lambda img: gr.update(visible=img is not None), inputs=inpaint_input_image, outputs=describe_inpaint_button, show_progress=False, queue=False)

            describe_inpaint_button.click(trigger_auto_describe, inputs=[describe_methods, inpaint_input_image, prompt, describe_apply_styles, describe_output_tags, describe_output_chinese, describe_output_artist], outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(stash_preview_image, inputs=[enhance_input_image], outputs=[enhance_input_image, enhance_input_image_full], queue=False, show_progress=False) \
                .then(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(lambda: (gr.update(), gr.update()), inputs=[], outputs=[prompt, style_selections], show_progress=False, queue=False) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')
            enhance_input_image.change(lambda img: gr.update(visible=img is not None), inputs=enhance_input_image, outputs=describe_enhance_button, show_progress=False, queue=False)
            enhance_input_image.change(lambda img, full: None if img is None else full, inputs=[enhance_input_image, enhance_input_image_full], outputs=[enhance_input_image_full], show_progress=False, queue=False)

            describe_enhance_button.click(trigger_auto_describe, inputs=[describe_methods, enhance_input_image_full, prompt, describe_apply_styles, describe_output_tags, describe_output_chinese, describe_output_artist], outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

    note_box_outputs = [params_note_info, params_note_close_button, params_note_input_name, params_note_delete_button, params_note_regen_button, params_note_preset_button, params_note_box, state_topbar]

    prompt_delete_button.click(toolbox.toggle_note_box_delete, inputs=state_topbar, outputs=note_box_outputs, show_progress=False)
    params_note_delete_button.click(toolbox.delete_image, inputs=state_topbar, outputs=[gallery, gallery_index, params_note_delete_button, params_note_box, gallery_index_stat], show_progress=False) \
            .then(toolbox.close_note_box, inputs=state_topbar, outputs=note_box_outputs, show_progress=False) \
            .then(lambda x: None, inputs=gallery_index_stat, queue=False, show_progress=False, _js='(x)=>{refresh_finished_images_catalog_label(x);}')
    
    prompt_regen_button.click(toolbox.toggle_note_box_regen, inputs=model_check + [state_topbar], outputs=note_box_outputs, show_progress=False)
    params_note_regen_button.click(toolbox.reset_image_params, inputs=[state_topbar, state_is_generating, inpaint_mode], outputs=reset_preset_layout + reset_preset_func + scene_frontend_ctrls + load_data_outputs + [params_note_regen_button, params_note_box], show_progress=False) \
            .then(toolbox.close_note_box, inputs=state_topbar, outputs=note_box_outputs, show_progress=False)
    prompt_preset_button.click(toolbox.toggle_note_box_preset, inputs=model_check + [state_topbar], outputs=note_box_outputs, show_progress=False)
    scene_prompt_preset_button.click(toolbox.toggle_note_box_preset, inputs=model_check + [state_topbar], outputs=note_box_outputs, show_progress=False)
    params_note_close_button.click(toolbox.close_note_box, inputs=state_topbar, outputs=note_box_outputs, show_progress=False)
    params_note_preset_button.click(toolbox.save_preset, inputs=[params_note_input_name, params_backend, state_topbar] + reset_preset_func + load_data_outputs + scene_preset_save_ctrls, outputs=[params_note_input_name, params_note_preset_button, params_note_box, preset_store_list] + nav_bars + [system_params], show_progress=False) \
        .then(toolbox.preset_store_unmount, inputs=state_topbar, outputs=preset_store_list, show_progress=False, queue=False) \
        .then(toolbox.preset_store_mount, inputs=state_topbar, outputs=preset_store_list, show_progress=False, queue=False) \
        .then(toolbox.close_note_box, inputs=state_topbar, outputs=note_box_outputs, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')

    def _sanitize_ip_types(*values):
        allowed = [flags.cn_canny, flags.cn_cpds, flags.cn_pose]
        fallback = allowed[0]
        return [v if v in allowed else fallback for v in values]

    
    after_identity = [gallery_index, index_radio, gallery_index_stat, layer_method, layer_input_image, preset_store, preset_store_list, history_link, identity_introduce, configure_panel, local_system_tab, admin_panel, p2p_panel, admin_link, system_params] + ip_types
    identity_phrases_confirm_button.click(lambda a, b, c: simpleai.set_phrases(a,b,c,'confirm'), inputs=identity_input_info + [identity_phrase_input], outputs=identity_ctrls + [current_id_info, current_upstream_status, identity_export_btn], show_progress=False) \
        .then(topbar.update_after_identity_all, inputs=state_topbar, outputs=nav_bars + after_identity + user_app_ctrls, show_progress=False) \
        .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
        .then(wildcards.refresh_wildcards_components, inputs=state_topbar, outputs=[wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False) \
        .then(_qwen_refresh_style_preset_dropdowns, inputs=[state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices], queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')
    identity_confirm_button.click(simpleai.confirm_identity, inputs=identity_input_info + [identity_phrase_input], outputs=identity_ctrls + [current_id_info, current_upstream_status, identity_export_btn], show_progress=False) \
        .then(topbar.update_after_identity_all, inputs=state_topbar, outputs=nav_bars + after_identity + user_app_ctrls, show_progress=False) \
        .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
        .then(wildcards.refresh_wildcards_components, inputs=state_topbar, outputs=[wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False) \
        .then(_qwen_refresh_style_preset_dropdowns, inputs=[state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices], queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')
    identity_unbind_button.click(simpleai.unbind_identity, inputs=identity_input_info + [identity_phrase_input], outputs=identity_ctrls + identity_input + [current_id_info, current_upstream_status, identity_export_btn], show_progress=False) \
        .then(topbar.update_after_identity_all, inputs=state_topbar, outputs=nav_bars + after_identity + user_app_ctrls, show_progress=False) \
        .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
        .then(wildcards.refresh_wildcards_components, inputs=state_topbar, outputs=[wildcards_list, wc_name, wildcard_tag_name_selection], show_progress=False, queue=False) \
        .then(_qwen_refresh_style_preset_dropdowns, inputs=[state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices], queue=False, show_progress=False) \
        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')
    binding_id_button.click(simpleai.toggle_identity_dialog, inputs=state_topbar, outputs=[identity_dialog, current_id_info, current_upstream_status, identity_export_btn] + identity_ctrls + identity_input, show_progress=False)

    p2p_active_checkbox.change(simpleai.toggle_p2p, inputs=[p2p_active_checkbox, state_topbar], outputs=[p2p_active_checkbox, p2p_remote_process, p2p_ping_btn]) \
                        .then(topbar.update_after_identity, inputs=state_topbar, outputs=nav_bars + after_identity, show_progress=False) \
                        .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
                        .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}')

    reset_layout_params = nav_bars + reset_preset_layout + reset_preset_func + scene_frontend_ctrls + load_data_outputs + after_identity
    reset_layout_ui_outputs = nav_bars + reset_preset_layout + reset_preset_func + scene_frontend_ctrls
    reset_layout_values_outputs = load_data_outputs + after_identity + \
                                  [scene_canvas_image, scene_input_image1, scene_input_image2, scene_lora_model, scene_lora_model_2, scene_lora_model_3, scene_lora_model_4, scene_use_lora, quick_enhance, model_gallery, gallery_visible, current_previews, active_target, base_preview_btn, refiner_preview_btn] + \
                                  lora_galleries + lora_gallery_visible + lora_current_previews + lora_preview_btns

    topbar.reset_layout_num = len(reset_layout_ui_outputs) - len(nav_bars)
    topbar.reset_layout_ui_outputs_len = len(reset_layout_ui_outputs)
    reset_preset_inputs = [prompt, negative_prompt, state_topbar, state_is_generating, inpaint_mode, comfyd_active_checkbox]
    reset_values_inputs = [state_topbar, state_is_generating, inpaint_mode, use_resolution_override_checkbox]

    for i in range(shared.BUTTON_NUM):
        bar_buttons[i].click(topbar.reset_layout_ui, inputs=reset_preset_inputs + [bar_buttons[i]], outputs=reset_layout_ui_outputs + [state_topbar, comparison_state, comparison_box, progress_gallery, compare_btn, progress_window], queue=False, show_progress=False) \
               .then(lambda sp, umf: refresh_files_clicked(sp, umf, False), inputs=[state_topbar, model_filter_state], outputs=refresh_files_output + lora_ctrls, queue=False, show_progress=False) \
               .then(topbar.reset_layout_values, inputs=reset_values_inputs, outputs=reset_layout_values_outputs, show_progress=False) \
               .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
               .then(lambda: True, inputs=[], outputs=[scene_to_main_sync_lock], queue=False, show_progress=False) \
               .then(sync_scene_model_selections, inputs=[state_topbar, base_model, refiner_model] + lora_ctrls, outputs=[scene_base_model, scene_refiner_model, scene_use_lora, lora_group, scene_lora_model, scene_lora_weight, scene_lora_model_2, scene_lora_weight_2, scene_lora_model_3, scene_lora_weight_3, scene_lora_model_4, scene_lora_weight_4], queue=False, show_progress=False) \
               .then(lambda: False, inputs=[], outputs=[scene_to_main_sync_lock], queue=False, show_progress=False) \
               .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x); refresh_style_localization(); refresh_scene_localization();}') \
               .then(update_describe_output_tags, inputs=engine_class_display, outputs=describe_output_tags, queue=False, show_progress=False) \
               .then(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state, outpaint_selections, state_topbar], outputs=[inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field], show_progress=False, queue=False) \
               .then(check_camera_control_visibility, inputs=[scene_theme, state_topbar], outputs=[camera_control_accordion, anglelight_control_accordion, style_transfer_accordion, sam3_video_mask_accordion], queue=False, show_progress=False) \
               .then(inpaint_engine_state_change, inputs=[inpaint_engine_state, state_topbar] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)  \
               .then(check_and_show_missing_models, inputs=[bar_buttons[i], state_topbar], outputs=[missing_model_modal, missing_model_list, missing_model_total_progress, missing_model_btn]) \
               .then(topbar.stop_comfyd_background, inputs=[comfyd_active_checkbox], queue=False)
    shared.gradio_root.load(fn=lambda x: x, inputs=system_params, outputs=state_topbar, _js=topbar.get_system_params_js, queue=False, show_progress=False) \
                      .then(topbar.init_nav_bars, inputs=[state_topbar] + admin_ctrls, outputs=[progress_window, language_ui, background_theme, preset_instruction] + user_app_ctrls + admin_ctrls, show_progress=False) \
                      .then(_qwen_refresh_style_preset_dropdowns, inputs=[state_topbar, qwen_design_style_preset_choices, qwen_custom_style_preset_choices], outputs=[qwen_design_style_preset_choices, qwen_custom_style_preset_choices], queue=False, show_progress=False) \
                      .then(topbar.reset_layout_ui, inputs=reset_preset_inputs, outputs=reset_layout_ui_outputs + [state_topbar, comparison_state, comparison_box, progress_gallery, compare_btn, progress_window], show_progress=False) \
                      .then(lambda sp, umf: refresh_files_clicked(sp, umf, False), inputs=[state_topbar, model_filter_state], outputs=refresh_files_output + lora_ctrls, queue=True, show_progress=False) \
                      .then(topbar.refresh_preset_store_list, inputs=state_topbar, outputs=preset_store_list, show_progress=False, queue=False) \
                      .then(topbar.reset_layout_values, inputs=reset_values_inputs, outputs=reset_layout_values_outputs, show_progress=False) \
                      .then(_sanitize_ip_types, inputs=ip_types, outputs=ip_types, queue=False, show_progress=False) \
                      .then(lambda: True, inputs=[], outputs=[scene_to_main_sync_lock], queue=False, show_progress=False) \
                      .then(sync_scene_model_selections, inputs=[state_topbar, base_model, refiner_model] + lora_ctrls, outputs=[scene_base_model, scene_refiner_model, scene_use_lora, lora_group, scene_lora_model, scene_lora_weight, scene_lora_model_2, scene_lora_weight_2, scene_lora_model_3, scene_lora_weight_3, scene_lora_model_4, scene_lora_weight_4], queue=False, show_progress=False) \
                      .then(lambda: False, inputs=[], outputs=[scene_to_main_sync_lock], queue=False, show_progress=False) \
                      .then(fn=lambda x: None, inputs=system_params, _js='(x)=>{refresh_topbar_status_js(x);}') \
                      .then(topbar.sync_message, inputs=state_topbar) \
                      .then(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state, outpaint_selections, state_topbar], outputs=[inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field], show_progress=False, queue=False) \
                      .then(check_camera_control_visibility, inputs=[scene_theme, state_topbar], outputs=[camera_control_accordion, anglelight_control_accordion, style_transfer_accordion, sam3_video_mask_accordion], queue=False, show_progress=False) \
                      .then(lambda x: x, inputs=aspect_ratios_selections[0], outputs=aspect_ratios_selection, queue=False, show_progress=False) \
                      .then(lambda x: None, inputs=aspect_ratios_selections[0], queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}') \
                      .then(fn=lambda: None, _js='refresh_grid_delayed') \
                      .then(fn=lambda: [update_trigger_word(lora_models[i].value) for i in range(len(lora_models))], inputs=[], outputs=lora_trigger_words, queue=False, show_progress=False) \
                      .then(fn=lambda: None, _js='bindPluginBtn')

def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)

#dump_default_english_config()
import logging
import httpx
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if ads.get_admin_default('comfyd_active_checkbox') and not args_manager.args.disable_comfyd and not args_manager.args.disable_backend:
    comfyd.active(True)
if ads.get_admin_default('p2p_active_checkbox'):
    if shared.upstream_did:
        shared.token.p2p_start()
    else:
        shared.upstream_did = shared.token.get_p2p_upstream_did()
    if shared.upstream_did:
        shared.upstream_did = f'{shared.upstream_did}:P2P'
if ads.get_admin_default('p2p_remote_process'):
    p2p_task.init_p2p_task(worker, model_management, shared.token, minicpm)

# Fix for global proxy issues causing "Expecting value: line 1 column 1"
for key in ['NO_PROXY', 'no_proxy']:
    current_val = os.environ.get(key, '')
    if 'localhost' not in current_val:
        os.environ[key] = f"localhost,127.0.0.1,0.0.0.0,{current_val}".strip(',')

import socket
import psutil

current_listen = args_manager.args.listen

is_listen_invalid = (
    current_listen is None or
    current_listen == "0.0.0.0" or
    current_listen == "127.0.0.1" or
    simpleai.is_fake_or_suspicious_ip(current_listen)
)

if is_listen_invalid:
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        if simpleai.is_fake_or_suspicious_ip(local_ip) or simpleai.is_fake_or_suspicious_ip(current_listen):
            logging.warning(f"Detected Fake/Proxy IP configuration (Listen: {current_listen}, Resolved: {local_ip}).")
            best_ip = simpleai.get_best_local_ip()
            if best_ip != '127.0.0.1':
                logging.info(f"Forcing Gradio to bind to valid LAN IP: {best_ip}")
                args_manager.args.listen = best_ip
                
                # Re-check port availability because IP has changed
                if not simpleai.is_port_available(args_manager.args.port, best_ip):
                    new_port = simpleai.find_available_port(args_manager.args.port, host=best_ip, suppress_logging=True)
                    if new_port != args_manager.args.port:
                        logging.info(f"Port {args_manager.args.port} is occupied on {best_ip}, automatically switched to: {new_port}")
                        args_manager.args.port = new_port
            else:
                logging.info(f"Could not find a better LAN IP, falling back to 0.0.0.0 to ensure accessibility.")
                args_manager.args.listen = "0.0.0.0"
                
                # Re-check for 0.0.0.0 as well
                if not simpleai.is_port_available(args_manager.args.port, "0.0.0.0"):
                    new_port = simpleai.find_available_port(args_manager.args.port, host="0.0.0.0", suppress_logging=True)
                    if new_port != args_manager.args.port:
                        logging.info(f"Port {args_manager.args.port} is occupied on 0.0.0.0, automatically switched to: {new_port}")
                        args_manager.args.port = new_port
    except Exception as e:
        if simpleai.is_fake_or_suspicious_ip(current_listen):
             args_manager.args.listen = "0.0.0.0"
             
             # Re-check for 0.0.0.0
             if not simpleai.is_port_available(args_manager.args.port, "0.0.0.0"):
                 new_port = simpleai.find_available_port(args_manager.args.port, host="0.0.0.0", suppress_logging=True)
                 if new_port != args_manager.args.port:
                     logging.info(f"Port {args_manager.args.port} is occupied on 0.0.0.0, automatically switched to: {new_port}")
                     args_manager.args.port = new_port
        pass

app, local_url, share_url = shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    root_path=args_manager.args.webroot,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[
        modules.config.path_userhome,
        modules.config.get_path_models_root(),
        *modules.config.paths_checkpoints,
        *modules.config.paths_loras
    ],
    blocked_paths=[constants.AUTH_FILENAME],
    prevent_thread_lock=True
)

import threading
from fastapi import Body
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool
import enhanced.layerforge_matting as layerforge_matting
import enhanced.layerforge_openpose as layerforge_openpose

_matting_lock = threading.Lock()
_openpose_lock = threading.Lock()

@app.get("/matting/check-model")
async def matting_check_model():
    return layerforge_matting.check_model_availability()

@app.post("/matting")
async def matting_endpoint(payload: dict = Body(...)):
    try:
        image_data = payload.get("image")
        threshold = payload.get("threshold", 0.5)
        if not isinstance(image_data, str) or not image_data.startswith("data:image"):
            return JSONResponse(
                {
                    "error": "Bad Request",
                    "details": "Missing or invalid 'image' data URL.",
                },
                status_code=400,
            )

        def safe_process():
            with _matting_lock:
                return layerforge_matting.process_matting(image_data, threshold)

        result = await run_in_threadpool(safe_process)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "error": "Matting Error",
                "details": str(e),
            },
            status_code=500,
        )

@app.get("/openpose/check-model")
async def openpose_check_model():
    return layerforge_openpose.check_model_availability()

@app.post("/openpose/detect")
async def openpose_detect_endpoint(payload: dict = Body(...)):
    try:
        image_data = payload.get("image")
        detect_resolution = payload.get("detect_resolution", 512)
        allow_download = payload.get("allow_download", False)
        if not isinstance(image_data, str) or not image_data.startswith("data:image"):
            return JSONResponse(
                {
                    "error": "Bad Request",
                    "details": "Missing or invalid 'image' data URL.",
                },
                status_code=400,
            )

        def safe_process():
            with _openpose_lock:
                return layerforge_openpose.process_openpose(image_data, detect_resolution, allow_download)

        result = await run_in_threadpool(safe_process)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "error": "OpenPose Error",
                "details": str(e),
            },
            status_code=500,
        )

@app.get("/wildcards/readme")
async def wildcards_readme():
    import html
    try:
        md_path = os.path.join(os.path.dirname(__file__), "wildcards", "readme.md")
        content = open(md_path, encoding="utf-8").read()
        body = html.escape(content)
        page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>wildcards/readme.md</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 16px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div><a href="readme/raw" target="_blank" rel="noopener noreferrer">Open raw</a></div>
  <pre>{body}</pre>
</body>
</html>"""
        return HTMLResponse(content=page)
    except Exception as e:
        return PlainTextResponse(content=str(e), status_code=500)

@app.get("/wildcards/readme/raw")
async def wildcards_readme_raw():
    try:
        md_path = os.path.join(os.path.dirname(__file__), "wildcards", "readme.md")
        content = open(md_path, encoding="utf-8").read()
        return PlainTextResponse(content=content)
    except Exception as e:
        return PlainTextResponse(content=str(e), status_code=500)

threading.Event().wait()
