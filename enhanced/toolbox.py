import os
import json
import copy
import re
import math
import time
import gradio as gr
import modules.config as config
import modules.sdxl_styles as sdxl_styles
import enhanced.all_parameters as ads
import enhanced.topbar as topbar
import enhanced.gallery as gallery
import enhanced.version as version
import modules.flags as flags
import modules.meta_parser as meta_parser
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enhanced.simpleai import sync_model_info, get_path_in_user_dir
from modules.model_loader import load_file_from_url
from shared import sysinfo


# app context
toolbox_note_preset_title='Save a new preset for the current params and configuration.'
toolbox_note_regenerate_title='Extract parameters to backfill for regeneration. Please note that some parameters will be modified!'
toolbox_note_embed_title='Embed parameters into images for easy identification of image sources and communication and learning.'
toolbox_note_missing_muid='The model in the params and configuration is missing MUID. And the system will spend some time calculating the hash of model files and synchronizing information to obtain the muid for usability and transferability.'

def make_infobox_markdown(info, theme):
    bgcolor = '#ddd'
    if theme == "dark":
        bgcolor = '#444'
    # 为 div 添加 padding，特别是右侧留出空间给关闭按钮 (×)
    html = f'<div style="background: {bgcolor}; padding: 10px 35px 10px 15px; border-radius: 8px;">'
    if info:
        for key in info:
            if key in ['Filename', 'Advanced_parameters', 'Fooocus V2 Expansion', 'Metadata Scheme', 'Version', 'Upscale (Fast)'] or info[key] in [None, '', 'None']:
                continue
            html += f'<b>{key}:</b> {info[key]}<br/>'
    else:
        html += '<p>info</p>'
    html += '</div>'
    return html


def toggle_toolbox(state, state_params):
    if "gallery_state" in state_params and state_params["gallery_state"] == 'finished_index':
        return [gr.update(visible=state)]
    else:
        return [gr.update(visible=False)] 


def toggle_prompt_info(state_params):
    infobox_state = state_params.get("infobox_state", False)
    infobox_state = not infobox_state
    state_params.update({"infobox_state": infobox_state})

    prompt_info_data = state_params.get("prompt_info")
    if not prompt_info_data or not isinstance(prompt_info_data, list) or len(prompt_info_data) < 2:
        output_list = state_params.get("__output_list", [])
        if output_list:
            prompt_info_data = [output_list[0], 0]
            state_params.update({"prompt_info": prompt_info_data})
        else:
            prompt_info_data = [None, 0]

    [choice, selected] = prompt_info_data
    prompt_info = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"], user_did=state_params["user"].get_did())
    return (
        gr.update(value=make_infobox_markdown(prompt_info, state_params['__theme']), visible=infobox_state),
        gr.update(visible=infobox_state),
        gr.update(visible=infobox_state),
        state_params
    )

def close_prompt_info(state_params):
    state_params.update({"infobox_state": False})
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        state_params
    )


def check_preset_models(checklist, state_params):
    note_box_state = state_params["note_box_state"]
    note_box_state[2] = 0
    #for i in range(len(checklist)):
    #    if checklist[i] and checklist[i] != 'None':
    #        k1 = "checkpoints/"+checklist[i]
    #        k2 = "loras/"+checklist[i]
    #        if (i<2 and (k1 not in models_info.keys() or not models_info[k1]['muid'])) or (i>=2 and (k2 not in models_info.keys() or not models_info[k2]['muid'])):
    #            note_box_state[2] = 1
    #            break
    state_params.update({"note_box_state": note_box_state})
    return state_params


def toggle_note_box(item, state_params):
    note_box_state = state_params["note_box_state"]
    if note_box_state[0] is None:
        note_box_state[0] = item
    if item == note_box_state[0]:
        note_box_state[1] = not note_box_state[1]
    elif not note_box_state[1]:
        note_box_state[1] = not note_box_state[1]
        note_box_state[0] = item
    else:
        note_box_state[0] = item
        note_box_state[1] = True

    state_params.update({"note_box_state": note_box_state})
    flag = note_box_state[1]
    title_extra = ""
    if note_box_state[2]:
        title_extra = '\n' # + toolbox_note_missing_muid

    info_val = ""
    if item == 'delete':
        info_val = f'DELETE the image from output directory and logs!'
    elif item == 'regen':
        info_val = toolbox_note_regenerate_title
    elif item == 'preset':
        info_val = toolbox_note_preset_title + title_extra

    # 返回顺序: info, close_btn, input_name, delete_btn, regen_btn, preset_btn, box, state_params
    return (
        gr.update(value=info_val, visible=flag),
        gr.update(visible=flag),
        gr.update(visible=flag if item == 'preset' else False),
        gr.update(visible=flag if item == 'delete' else False),
        gr.update(visible=flag if item == 'regen' else False),
        gr.update(visible=flag if item == 'preset' else False),
        gr.update(visible=flag),
        state_params
    )

def toggle_note_box_delete(state_params):
    return toggle_note_box('delete', state_params)


def toggle_note_box_regen(*args):
    args = list(args)
    state_params = args.pop()
    lora_count = len(config.default_loras)
    for i in range(lora_count):
        if len(args) > 4:
            del args[4]
            del args[4]
    checklist = args[2:]
    state_params = check_preset_models(checklist, state_params)
    return toggle_note_box('regen', state_params)

def toggle_note_box_preset(*args):
    args = list(args)
    state_params = args.pop()
    lora_count = len(config.default_loras)
    for i in range(lora_count):
        if len(args) > 4:
            del args[4]
            del args[4]
    checklist = args[2:]
    state_params = check_preset_models(checklist, state_params)
    return toggle_note_box('preset', state_params)


def close_note_box(state_params):
    state_params.update({"note_box_state": ['', 0, 0]})
    # 隐藏所有组件，返回顺序需与 webui.py 中的 outputs 一致
    return (
        gr.update(visible=False), # info
        gr.update(visible=False), # close_btn
        gr.update(visible=False), # input_name
        gr.update(visible=False), # delete_btn
        gr.update(visible=False), # regen_btn
        gr.update(visible=False), # preset_btn
        gr.update(visible=False), # box
        state_params
    )


filename_regex = re.compile(r'\<div id=\"(.*?)_png\"')

def delete_image(state_params):
    if 'engine_type' in state_params and state_params['engine_type'] == 'video':
        return [gr.update()] * 4 + [state_params['__finished_nums_pages']]

    [choice, selected] = state_params["prompt_info"]
    if choice is None and "__output_list" in state_params and len(state_params["__output_list"]) > 0:
        choice = state_params["__output_list"][0]
        state_params["prompt_info"][0] = choice

    max_per_page = state_params["__max_per_page"]
    max_catalog = state_params["__max_catalog"]
    user_did = state_params["user"].get_did()
    info = gallery.get_images_prompt(choice, selected, max_per_page, user_did=user_did)
    if not info or "Filename" not in info:
        logger.warning(f"Delete image failed: Image info not found for choice={choice}, selected={selected}")
        return [gr.update()] * 4 + [state_params['__finished_nums_pages']]
    file_name = info["Filename"]
    output_index = choice.split('/')
    user_path_outputs = config.get_user_path_outputs(user_did)
    dir_path = os.path.join(user_path_outputs, "20{}".format(output_index[0]))

    log_path = os.path.join(dir_path, 'log.html')
    if os.path.exists(log_path):
        file_text = ''
        d_line_flag = False
        with open(log_path, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                match = filename_regex.search(line)
                if match:
                    if match.group(1)==file_name[:-4]:
                        d_line_flag = True
                        line = log_file.readline()
                        continue
                    if d_line_flag:
                        d_line_flag = False
                if d_line_flag:
                    line = log_file.readline()
                    continue
                file_text += line
                line = log_file.readline()
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(file_text)
        logger.info(f'Delete item from log.html: {file_name}')

    log_name = os.path.join(dir_path, "log_ads.json")
    log_ext = {}
    if os.path.exists(log_name):
        log_ext = {}
        with open(log_name, "r", encoding="utf-8") as log_file:
            log_ext.update(json.load(log_file))
        if file_name in log_ext.keys():
            log_ext.pop(file_name)

        if not log_ext:
            os.remove(log_name)
        else:
            with open(log_name, 'w', encoding='utf-8') as log_file:
                json.dump(log_ext, log_file)

    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    logger.info(f'Delete image file: {file_path}')

    image_list_nums = len(gallery.refresh_images_catalog(output_index[0], True, user_did))
    if image_list_nums<=0:
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(log_name):
            os.remove(log_name)

        try:
            index = state_params["__output_list"].index(choice)
        except ValueError:
            index = 0

        output_list, finished_nums, finished_pages = gallery.refresh_output_list(max_per_page, max_catalog, user_did)
        state_params.update({"__output_list": output_list})
        state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
        if index>= len(state_params["__output_list"]):
            index = len(state_params["__output_list"]) -1
            if index<0:
                index = 0
        choice = None if len(output_list)==0 else state_params["__output_list"][index]
    elif image_list_nums < max_per_page:
        if selected > image_list_nums-1:
            selected = image_list_nums-1
        finished_nums_pages = state_params["__finished_nums_pages"]
        finished_nums = int(finished_nums_pages.split(',')[0])-1
        finished_pages = finished_nums_pages.split(',')[1]
        state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
    else:
        if image_list_nums % max_per_page == 0:
            page = int(output_index[1])
            if page > image_list_nums//max_per_page:
                page = image_list_nums//max_per_page
            if page == 1:
                choice = output_index[0]
            else:
                choice = output_index[0] + '/' + str(page)
            output_list, finished_nums, finished_pages = gallery.refresh_output_list(max_per_page, max_catalog, user_did)
            state_params.update({"__output_list": output_list})
            state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
        else:
            finished_nums_pages = state_params["__finished_nums_pages"]
            finished_nums = int(finished_nums_pages.split(',')[0])-1
            finished_pages = finished_nums_pages.split(',')[1]
            state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})

    state_params.update({"prompt_info":[choice, selected]})
    images_gallery = gallery.get_images_from_gallery_index(choice, max_per_page, user_did)
    state_params.update({"note_box_state": ['',0,0]})
    return gr.update(value=images_gallery), gr.update(choices=state_params["__output_list"], value=choice, visible=True if choice else False), gr.update(visible=False), gr.update(visible=False), state_params['__finished_nums_pages']


def reset_params_by_image_meta(metadata, state_params, is_generating, inpaint_mode):
    if metadata is None:
        metadata = {}
    metadata_scheme = meta_parser.MetadataScheme('simple')
    metadata_parser = meta_parser.get_metadata_parser(metadata_scheme)
    parsed_parameters = metadata_parser.to_json(metadata)

    #config_preset = config.try_get_preset_content(state_params["__preset"], state_params["user"].get_did())
    #preset_prepared = meta_parser.parse_meta_from_preset(config_preset)
    #if "engine" in preset_prepared:
    #    parsed_parameters.update({"engine": preset_prepared["engine"]})
    
    results = meta_parser.switch_layout_template(parsed_parameters, state_params)
    results += meta_parser.load_parameter_button_click(parsed_parameters, is_generating, inpaint_mode)

    engine_name = parsed_parameters.get("Backend Engine", parsed_parameters.get("backend_engine", "SDXL-Fooocus"))
    logger.info(f'Reset_params_from_image: -->{engine_name} params from the image with embedded parameters.')
    return results

def reset_image_params(state_params, is_generating, inpaint_mode):
    [choice, selected] = state_params["prompt_info"]
    metainfo = gallery.get_images_prompt(choice, selected, state_params["__max_per_page"], user_did=state_params["user"].get_did())
    if metainfo is None:
        metainfo = {}
    metadata = copy.deepcopy(metainfo)
    metadata['Refiner Model'] = metainfo.get('Refiner Model', 'None')
    state_params.update({"note_box_state": ['',0,0]})

    results = reset_params_by_image_meta(metadata, state_params, is_generating, inpaint_mode)
    return results + [gr.update(visible=False)] * 2


def save_preset(*args):    
    args = list(args)
    args.reverse()
    name = args.pop()
    backend_params = dict(args.pop())
    state_params = dict(args.pop())

    output_format = args.pop()
    inpaint_advanced_masking_checkbox = args.pop()
    mixing_image_prompt_and_vary_upscale = args.pop()
    mixing_image_prompt_and_inpaint = args.pop()
    backfill_prompt = args.pop()
    translation_methods = args.pop()
    input_image_checkbox = args.pop()
    quick_enhance = args.pop()

    progress_video = args.pop()
    progress_gallery = args.pop()
    progress_window = args.pop()
    gallery = args.pop()
    gallery_index = args.pop()

    image_number = int(args.pop())
    prompt = args.pop()
    negative_prompt = args.pop()
    style_selections = args.pop()
    performance_selection = args.pop()
    overwrite_step = int(args.pop())
    overwrite_switch = args.pop()
    aspect_ratios_selection = args.pop()
    overwrite_width = args.pop()
    overwrite_height = args.pop()
    guidance_scale = args.pop()
    sharpness = args.pop()
    adm_scaler_positive = args.pop()
    adm_scaler_negative = args.pop()
    adm_scaler_end = args.pop()
    refiner_swap_method = args.pop()
    adaptive_cfg = args.pop()
    clip_skip = args.pop()
    base_model = args.pop()
    refiner_model = args.pop()
    refiner_switch = args.pop()
    sampler_name = args.pop()
    scheduler_name = args.pop()
    vae_name = args.pop()
    seed_random = args.pop()
    image_seed = args.pop()
    inpaint_engine = args.pop()
    inpaint_engine_state = args.pop()
    inpaint_mode = args.pop()
    enhance_inpaint_mode_ctrls = [args.pop() for _ in range(config.default_enhance_tabs)]
    #generate_button = args.pop()
    #load_parameter_button = args.pop()
    freeu_ctrls = [bool(args.pop()), float(args.pop()), float(args.pop()), float(args.pop()), float(args.pop())]
    loras = [(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in range(config.default_max_lora_number)]
    loras = [[n, w] for (f, n, w) in loras]
    enhance_checkbox = args.pop()
    enhance_enabled_1 = args.pop()
    enhance_enabled_2 = args.pop()
    enhance_enabled_3 = args.pop()
    enhance_uov_method = args.pop()
    enhance_uov_strength = args.pop()

    scene_theme = None
    scene_additional_prompt = None
    scene_additional_prompt_2 = None
    scene_var_number = None
    scene_var_number2 = None
    scene_var_number3 = None
    scene_var_number4 = None
    scene_var_number5 = None
    scene_var_number6 = None
    scene_var_number7 = None
    scene_var_number8 = None
    scene_var_number9 = None
    scene_var_number10 = None
    scene_steps = None
    scene_switch_option1 = None
    scene_switch_option2 = None
    scene_switch_option3 = None
    scene_switch_option4 = None
    scene_aspect_ratio = None
    scene_image_number = None
    scene_mask_color = None
    scene_use_lora = None

    if args:
        scene_theme = args.pop()
        scene_additional_prompt = args.pop()
        scene_additional_prompt_2 = args.pop()
        scene_var_number = args.pop()
        scene_var_number2 = args.pop()
        scene_var_number3 = args.pop()
        scene_var_number4 = args.pop()
        scene_var_number5 = args.pop()
        scene_var_number6 = args.pop()
        scene_var_number7 = args.pop()
        scene_var_number8 = args.pop()
        scene_var_number9 = args.pop()
        scene_var_number10 = args.pop()
        scene_steps = args.pop()
        scene_switch_option1 = args.pop()
        scene_switch_option2 = args.pop()
        scene_switch_option3 = args.pop()
        scene_switch_option4 = args.pop()
        scene_aspect_ratio = args.pop()
        scene_image_number = args.pop()
        scene_mask_color = args.pop()
        scene_use_lora = args.pop()

    if name:
        preset = {}
        prepared_engine = None
        try:
            prepared = state_params.get("__preset_prepared", None)
            if isinstance(prepared, dict):
                prepared_engine = prepared.get("engine", None)
        except Exception:
            prepared_engine = None

        if isinstance(prepared_engine, dict):
            engine = copy.deepcopy(prepared_engine)
        else:
            engine = copy.deepcopy(config.default_engine) if isinstance(config.default_engine, dict) else {}

        backend_engine = backend_params.get("backend_engine", None) or state_params.get("backend_engine", None) or state_params.get("engine", None) or engine.get("backend_engine", None) or config.backend_engine
        if isinstance(backend_engine, str):
            backend_engine = backend_engine.strip()
        if not backend_engine:
            backend_engine = config.backend_engine
        engine["backend_engine"] = backend_engine

        task_method = backend_params.get("task_method", None) or state_params.get("task_method", None)
        if isinstance(task_method, str):
            task_method = task_method.strip()
        else:
            task_method = None

        if task_method:
            if isinstance(engine.get("backend_params", None), dict):
                engine["backend_params"]["task_method"] = task_method
            else:
                engine["backend_params"] = {"task_method": task_method}

        engine_type = state_params.get("engine_type", None)
        if isinstance(engine_type, str) and engine_type:
            engine["engine_type"] = engine_type

        if scene_theme is not None and isinstance(state_params.get("scene_frontend", None), dict):
            scene_frontend = copy.deepcopy(state_params.get("scene_frontend", {}))
            old_themes = scene_frontend.get("theme", [])
            if not isinstance(old_themes, list):
                old_themes = []

            scene_frontend["theme"] = [scene_theme]

            for key, value in list(scene_frontend.items()):
                if not isinstance(value, dict):
                    continue
                if not any(t in value for t in old_themes):
                    continue
                chosen = value.get(scene_theme, next(iter(value.values()), None))
                if chosen is not None:
                    scene_frontend[key] = {scene_theme: chosen}

            task_method_map = scene_frontend.get("task_method", None)
            if isinstance(task_method_map, dict):
                chosen_task = task_method_map.get(scene_theme, next(iter(task_method_map.values()), None))
            else:
                chosen_task = None
            if not isinstance(chosen_task, str) or not chosen_task.strip():
                chosen_task = task_method
                if isinstance(chosen_task, str) and chosen_task.startswith("scene_"):
                    chosen_task = chosen_task[6:]
            if isinstance(chosen_task, str) and chosen_task.strip():
                scene_frontend["task_method"] = {scene_theme: chosen_task.strip()}

            def _set_theme_value(k, v):
                if v is None:
                    return
                scene_frontend[k] = {scene_theme: v}

            _set_theme_value("additional_prompt", scene_additional_prompt)
            _set_theme_value("additional_prompt_2", scene_additional_prompt_2)
            _set_theme_value("var_number", scene_var_number)
            _set_theme_value("var_number2", scene_var_number2)
            _set_theme_value("var_number3", scene_var_number3)
            _set_theme_value("var_number4", scene_var_number4)
            _set_theme_value("var_number5", scene_var_number5)
            _set_theme_value("var_number6", scene_var_number6)
            _set_theme_value("var_number7", scene_var_number7)
            _set_theme_value("var_number8", scene_var_number8)
            _set_theme_value("var_number9", scene_var_number9)
            _set_theme_value("var_number10", scene_var_number10)
            _set_theme_value("scene_steps", scene_steps)
            _set_theme_value("switch_option1", scene_switch_option1)
            _set_theme_value("switch_option2", scene_switch_option2)
            _set_theme_value("switch_option3", scene_switch_option3)
            _set_theme_value("switch_option4", scene_switch_option4)
            _set_theme_value("image_number", scene_image_number)
            _set_theme_value("mask_color", scene_mask_color)
            _set_theme_value("use_lora", scene_use_lora)

            def _normalize_aspect_ratio_to_raw(ar):
                if not isinstance(ar, str):
                    return None
                if "|" not in ar:
                    return ar
                left, ratio = ar.split("|", 1)
                if "×" in left:
                    width = left.split("×", 1)[0]
                    return f"{width}|{ratio}"
                return ar

            if scene_aspect_ratio is not None:
                selected_raw = _normalize_aspect_ratio_to_raw(scene_aspect_ratio)
                if selected_raw:
                    base_ar = scene_frontend.get("aspect_ratio", [])
                    if isinstance(base_ar, dict):
                        base_ar = base_ar.get(scene_theme, next(iter(base_ar.values()), []))
                    if not isinstance(base_ar, list):
                        base_ar = []
                    scene_frontend["aspect_ratio"] = [selected_raw] + [x for x in base_ar if x != selected_raw]

            engine["scene_frontend"] = scene_frontend

        backend_params_sanitized = engine.get("backend_params", None)
        if isinstance(backend_params_sanitized, dict):
            for k in [
                "nickname",
                "user_did",
                "translation_methods",
                "backfill_prompt",
                "comfyd_active_checkbox",
                "backend_engine",
            ]:
                backend_params_sanitized.pop(k, None)
            if not backend_params_sanitized:
                engine.pop("backend_params", None)

        preset["default_engine"] = engine

        preset["default_model"] = base_model
        preset["default_refiner"] = refiner_model
        preset["default_refiner_switch"] = refiner_switch
        preset["default_loras"] = loras
        preset["default_cfg_scale"] = guidance_scale
        preset["default_sample_sharpness"] = sharpness
        preset["default_sampler"] = sampler_name
        preset["default_scheduler"] = scheduler_name
        preset["default_performance"] = performance_selection
        preset["default_prompt"] = prompt
        preset["default_prompt_negative"] = negative_prompt
        preset["default_styles"] = style_selections
        preset["default_aspect_ratio"] = aspect_ratios_selection.split(' ')[0].replace(u'\u00d7','*')
        if ads.default["adm_scaler_positive"] != adm_scaler_positive or ads.default["adm_scaler_negative"] != adm_scaler_negative \
                or ads.default["adm_scaler_end"] != adm_scaler_end:
            preset["default_adm_guidance"] = f'({adm_scaler_positive}, {adm_scaler_negative}, {adm_scaler_end})'
        if ads.default["freeu"]!=freeu_ctrls[1:]:
            preset["default_freeu"]=freeu_ctrls[1:]
        if ads.default["adaptive_cfg"] != adaptive_cfg:
            preset["default_cfg_tsnr"] = adaptive_cfg
        if ads.default["overwrite_step"] != overwrite_step:
            preset["default_overwrite_step"] = overwrite_step
        if ads.default["overwrite_switch"] != overwrite_switch:
            preset["default_overwrite_switch"] = overwrite_switch
        if ads.default["inpaint_engine"] != inpaint_engine:
            preset["default_inpaint_engine"] = inpaint_engine
        if ads.default["clip_skip"] != clip_skip:
            preset["default_clip_skip"] = clip_skip
        if ads.default["vae"] != vae_name:
            preset["default_vae"] = vae_name
        if ads.default["overwrite_width"] != overwrite_width:
            preset["default_overwrite_width"] = overwrite_width
        if ads.default["overwrite_height"] != overwrite_height:
            preset["default_overwrite_height"] = overwrite_height
        if not seed_random:
            preset["default_image_seed"] = image_seed
        if ads.default.get("enhance_checkbox", False) != enhance_checkbox:
            preset["default_enhance_checkbox"] = enhance_checkbox
        if enhance_enabled_1:
            preset["default_enhance_enabled_1"] = enhance_enabled_1
        if enhance_enabled_2:
            preset["default_enhance_enabled_2"] = enhance_enabled_2
        if enhance_enabled_3:
            preset["default_enhance_enabled_3"] = enhance_enabled_3
        if ads.default.get("enhance_uov_method", 'upscale_15') != enhance_uov_method:
            preset["default_enhance_uov_method"] = enhance_uov_method
        if ads.default.get("enhance_uov_strength", 0.2) != enhance_uov_strength:
            preset["default_enhance_uov_strength"] = enhance_uov_strength

        preset["default_output_format"] = output_format
        preset["default_inpaint_advanced_masking"] = inpaint_advanced_masking_checkbox
        preset["default_mixing_image_prompt_and_vary_upscale"] = mixing_image_prompt_and_vary_upscale
        preset["default_mixing_image_prompt_and_inpaint"] = mixing_image_prompt_and_inpaint
        preset["default_backfill_prompt"] = backfill_prompt
        preset["default_translation_methods"] = translation_methods
        preset["default_input_image_checkbox"] = input_image_checkbox

        # preset["default_progress_video"] = progress_video
        # preset["default_progress_gallery"] = progress_gallery
        # preset["default_progress_window"] = progress_window

        preset["default_refiner_swap_method"] = refiner_swap_method
        preset["default_inpaint_engine_state"] = inpaint_engine_state
        preset["default_inpaint_mode"] = inpaint_mode
        preset["default_enhance_inpaint_mode_ctrls"] = enhance_inpaint_mode_ctrls

        preset["default_image_number"] = image_number

        preset["checkpoint_downloads"] = {}
        # if refiner_model and refiner_model != 'None':
        #     # preset["checkpoint_downloads"].update({refiner_model: get_muid_link("checkpoints/"+refiner_model)})

        preset["embeddings_downloads"] = {}
        prompt_tags = re.findall(r'[\(](.*?)[)]', negative_prompt) + re.findall(r'[\(](.*?)[)]', prompt)
        embeddings = {}
        for e in prompt_tags:
            embed = e.split(':')
            if len(embed)>2 and embed[0] == 'embedding':
                embeddings.update({embed[1]:embed[2]})
        embeddings = embeddings.keys()
        preset["embeddings_downloads"] = {} 


        m_dict = {}
        for key in style_selections:
            if key!='Fooocus V2':
                m_dict.update({key: sdxl_styles.styles[key]})
        if len(m_dict.keys())>0:
            preset["styles_definition"] = m_dict

        #logger.info(f'preset:{preset}')
        save_path = get_path_in_user_dir(name + '.json', state_params['user'].get_did(), catalog='presets')
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(preset, json_file, indent=4)

        logger.info(f'Saved the current params to {save_path}.')
    state_params.update({"note_box_state": ['',0,0]})
    cache_key = state_params['user'].get_did()
    topbar.preset_samples.pop(cache_key, None)
    topbar.preset_samples_user_mtime.pop(cache_key, None)
    topbar.preset_samples_base_mtime.pop(cache_key, None)
    topbar.preset_samples_complete_ts.pop(cache_key, None)
    results = [gr.update(visible=False)] * 3
    results += [gr.Dataset.update(samples=topbar.get_preset_samples(cache_key))]
    results += topbar.refresh_nav_bars(state_params)
    results += topbar.update_topbar_js_params(state_params)
    return results


def preset_store_unmount(state_params):
    return gr.update(samples=[], visible=False)


def preset_store_mount(state_params):
    user_did = state_params['user'].get_did() if 'user' in state_params and state_params['user'] else None
    return gr.update(samples=topbar.get_preset_samples(user_did), visible=True)


def sync_model_info_click(*args):

    downurls = list(args)
    #logger.info(f'downurls:{downurls} \nargs:{args}, len={len(downurls)}')
    keylist = sync_model_info(downurls)
    results = []
    nums = 0
    for k in keylist:
        muid = ' ' 
        durl = None 
        nums += 1 
        results += [gr.update(info=f'MUID={muid}', value=durl)]
    if nums:
        logger.info(f'There are {nums} model files missing MUIDs, which need to be added with download URLs before synchronizing.')
    return results

