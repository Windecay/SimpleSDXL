import os
import json
import hashlib
import gradio as gr
import numbers
import copy
import re
import args_manager
import random
import cv2
import numpy as np
import base64
import shared
import modules.util as util
import modules.config as config
import modules.flags
import modules.sdxl_styles
import modules.constants as constants
import modules.meta_parser as meta_parser
import modules.sdxl_styles as sdxl_styles
import modules.style_sorter as style_sorter
import enhanced.all_parameters as ads
import enhanced.gallery as gallery_util
import enhanced.superprompter as superprompter
import enhanced.comfy_task as comfy_task
import ldm_patched.modules.model_management
import logging
import threading
import time
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

from datetime import datetime
from modules.model_loader import load_file_from_url, is_models_file_absent, refresh_model_list, download_model_files
from modules.private_logger import get_current_html_path
from modules.meta_parser import get_welcome_image, describe_prompt_for_scene
from enhanced.simpleai import comfyd, get_path_in_user_dir, toggle_identity_dialog, sync_intput_reserved
from enhanced.minicpm import MiniCPM, minicpm
from simpleai_base.simpleai_base import export_identity_qrcode_svg, import_identity_qrcode, gen_ua_session

# app context
system_message = ''
config_ext = {}
enhanced_config = os.path.abspath(f'./enhanced/config.json')
if os.path.exists(enhanced_config):
    with open(enhanced_config, "r", encoding="utf-8") as json_file:
        config_ext.update(json.load(json_file))
else:
    config_ext.update({'fooocus_line': '# 2.1.852', 'simplesdxl_line': '# 2023-12-20'})

def preset_filter(presets):
    if not shared.gpu_arch:
        return presets

    try:
        compute_capability = int(shared.gpu_arch[2:])
        # 定义不同显卡系列的判断条件
        is_10_series_or_lower = compute_capability <= 61  # 10系列及以下
        is_20_series_or_lower = compute_capability <= 75  # 20系列及以下

        # 创建过滤后的预设列表
        filtered_presets = []
        seen_presets = set()
 
        # 获取是否跳过模型缺失过滤的设置
        missing_model_filter = ads.get_admin_default('missing_model_filter_checkbox')

        for preset_item in presets:
            # 标记是否应该被过滤
            should_filter = False
            filter_reason = ""

            # 获取预设名称字符串用于判断
            if isinstance(preset_item, list) and len(preset_item) > 0:
                preset_name = str(preset_item[0])
            else:
                # 处理普通字符串预设
                preset_name = str(preset_item)

            # 10系列及以下显卡过滤规则
            if is_10_series_or_lower:
                if 'fp4' in preset_name.lower() or 'int4' in preset_name.lower() or 'nun' in preset_name.lower():
                    should_filter = True
                    filter_reason = "10 Series GPU incompatible (Nunchaku)"

            elif is_20_series_or_lower:
                if ('NunQwen-Edit+' in preset_name) or ('fp4' in preset_name.lower()):
                    should_filter = True
                    filter_reason = "20 Series GPU incompatible (NunQwen-Edit+)"

            # 只有当启用模型缺失过滤选项时，才检查模型是否缺失
            if not should_filter and missing_model_filter and is_models_file_absent(preset_name, None):
                should_filter = True
                filter_reason = "Missing Model"

            # 记录被过滤的预置包和原因
            # if should_filter:
            #     logger.info(f"[Preset Filter] Disvisible: {preset_name}, Reason: {filter_reason}")

            if not should_filter:
                # 处理预设名称，去掉 _fp4 或 _int4 后缀
                if isinstance(preset_item, list) and len(preset_item) > 0:
                    original_name = str(preset_item[0])
                    # 检查并去掉 _fp4 或 _int4 后缀
                    if original_name.endswith('_fp4'):
                        modified_name = original_name[:-4]
                    elif original_name.endswith('_int4'):
                        modified_name = original_name[:-5]
                    else:
                        modified_name = original_name
                    if modified_name not in seen_presets:
                        seen_presets.add(modified_name)
                        modified_preset = [modified_name] + preset_item[1:]
                        filtered_presets.append(modified_preset)
                else:
                    # 处理普通字符串形式的预设
                    original_name = str(preset_item)
                    # 检查并去掉 _fp4 或 _int4 后缀
                    if original_name.endswith('_fp4'):
                        modified_name = original_name[:-4]
                    elif original_name.endswith('_int4'):
                        modified_name = original_name[:-5]
                    else:
                        modified_name = original_name
                    # 检查是否已经添加过相同名称的预设
                    if modified_name not in seen_presets:
                        seen_presets.add(modified_name)
                        filtered_presets.append(modified_name)
        return filtered_presets
    except Exception:
        return presets

def is_preset_file_allowed(p):
    if not p:
        return False
    if p.startswith('.'):
        return False
    p2 = str(p).replace('\\', '/').strip('/')
    parts = [x for x in p2.split('/') if x]
    if 'deprecated' in parts:
        return False
    if 'characters' in parts:
        return False
    return True

PRESET_MISSING_MARKER = "\u2B07"
PRESET_STORE_ORDER = [
    "Z-imageT",
    "Z-TTP",
    "Flux2-9BAIO",
    "Flux2-9BEdit",
    "Flux2AngleLight",
    "Flux2-9BA2R",
    "FluxAIO",
    "FluxAIOplus",
    "FluxKontext",
    "ClothingSwapplus",
    "NunClothingSwap_fp4",
    "NunClothingSwap_int4",
    "NunFlux_fp4",
    "NunFlux_int4",
    "NunQwen-Edit+_fp4",
    "NunQwen-Edit+_int4",
    "Qwen-Edit+",
    "Qwen2512",
    "QwenA2R",
    "QwenMultiAngle",
    "QwenNSFW",
    "Illustrious",
    "Illustrious2",
    "IllustriousAIO",
    "Anima",
    "NewBie",
    "Wan(I2V)",
    "Dasiwa(I2V)",
    "Wan(T2I)",
    "Wan-TTP",
    "Wan(T2V)",
    "Wan-Animate",
    "Animate-Outpaint",
    "Wan-SCAIL",
    "InfiniteTalk",
    "LTX2.3(IA2V)",
    "LTX2.3(TA2V)",
    "default",
    "eraser-a",
    "StyleTransfer",
    "StyleTransfer+",
    "x1-okremovebg",
    "x2-okimagerepair+",
    "OneKeyKontext",
    "OneKeyPose",
    "OneKey-Outpaint",
    "x3-swapface",
    "x4-okdepthstatue",
    "Tile",
    "relight",
    "SD15AIO",
]

def _strip_preset_marker(name):
    if not isinstance(name, str):
        return name
    base = name
    if base.endswith(PRESET_MISSING_MARKER):
        base = base[:-len(PRESET_MISSING_MARKER)].strip()
    return base

def _canonicalize_preset_name(name):
    if not isinstance(name, str):
        return name
    name = _strip_preset_marker(name)
    if name.endswith('.'):
        name = name[:-1]
    if name.endswith('_fp4'):
        name = name[:-4]
    elif name.endswith('_int4'):
        name = name[:-5]
    return name

def _append_status_marker(preset_name, user_did=None):
    if not isinstance(preset_name, str) or not preset_name:
        return preset_name
    base_name = _strip_preset_marker(preset_name).strip()
    if is_models_file_absent(base_name, user_did):
        return f"{base_name}{PRESET_MISSING_MARKER}"
    return base_name

def _apply_complete_markers(preset_list, user_did=None):
    marked_list = []
    for item in preset_list:
        if isinstance(item, list) and len(item) > 0:
            marked_name = _append_status_marker(item[0], user_did)
            marked_list.append([marked_name] + item[1:])
        else:
            marked_list.append(item)
    return marked_list

def _filter_existing_presets(presets, user_did=None):
    path_preset = os.path.abspath(f'./presets/')
    user_path_preset = get_path_in_user_dir('presets', user_did) if user_did else get_path_in_user_dir('presets')
    available = []
    arch_str = config.get_gpu_arch_str_in_preset_name()
    for preset in presets:
        preset_name = _strip_preset_marker(preset)
        if preset_name.endswith('.'):
            preset_file = os.path.join(user_path_preset, f'{preset_name[:-1]}.json')
            preset_file2 = os.path.join(user_path_preset, f'{preset_name[:-1]}{arch_str}.json')
        else:
            preset_file = os.path.join(path_preset, f'{preset_name}.json')
            preset_file2 = os.path.join(path_preset, f'{preset_name}{arch_str}.json')
        if os.path.exists(preset_file2):
            preset_file = preset_file2
        if os.path.exists(preset_file):
            available.append(preset)
    return available

def get_preset_name_list(user_session, ua_hash):
    user_did = shared.token.check_sstoken_and_get_did(user_session, ua_hash)
    is_guest = not user_did or shared.token.is_guest(user_did)
    if is_guest:
        try:
            presets_list = shared.token.get_local_vars("user_presets", "", user_session, ua_hash)
            if presets_list and presets_list not in ["Unknown", "None", "Default"]:
                return presets_list
        except Exception as e:
            print(f"[DEBUG] Error getting guest preset: {str(e)}")

    presets_list = shared.token.get_local_vars("user_presets", "", user_session, ua_hash)
    if presets_list and presets_list not in ["Unknown", "None", "Default"]:
        presets = [p.strip() for p in presets_list.split(',') if p.strip()]
        unique_presets = []
        for preset in presets:
            if preset not in unique_presets:
                unique_presets.append(preset)
        if config.preset in unique_presets:
            unique_presets.remove(config.preset)
        unique_presets.insert(0, config.preset)
        unique_presets = unique_presets[:shared.BUTTON_NUM]
        presets_list = ','.join(unique_presets)
        return presets_list

    if user_did and not shared.token.is_guest(user_did):
        user_preset_file = get_path_in_user_dir('presets.txt', user_did, 'presets')
        if not os.path.exists(user_preset_file):
            path_preset = os.path.abspath(f'./presets/')
            presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None)
                      if is_preset_file_allowed(p)]
            file_times = [(f[:-5], os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
            user_path_preset = get_path_in_user_dir('presets', user_did)
            if os.path.exists(user_path_preset):
                presets2 = [p for p in util.get_files_from_folder(user_path_preset, ['.json'], None)
                           if is_preset_file_allowed(p)]
                file_times2 = [(f'{f[:-5]}.', os.path.getmtime(os.path.join(user_path_preset, f))) for f in presets2]
                file_times = file_times + file_times2
            presets = sorted(file_times, key=lambda x: x[1], reverse=True)
            presets = [f[0] for f in presets]
            presets = preset_filter(presets)
            unique_presets = []
            for preset in presets:
                if preset not in unique_presets:
                    unique_presets.append(preset)
            presets = unique_presets
            if config.preset in presets:
                presets.remove(config.preset)
            presets.insert(0, config.preset)
            presets = presets[:shared.BUTTON_NUM]
            presets_list = ','.join(presets)
        else:
            with open(user_preset_file, 'r', encoding="utf-8") as nav_preset_file:
                presets_list = nav_preset_file.read()
    else:
        guest_preset_file = get_path_in_user_dir('presets.txt', catalog='presets')
        if os.path.exists(guest_preset_file):
            with open(guest_preset_file, 'r', encoding="utf-8") as nav_preset_file:
                presets_list = nav_preset_file.read()
        else:
            path_preset = os.path.abspath(f'./presets/')
            presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None)
                      if is_preset_file_allowed(p)]
            file_times = [(f[:-5], os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
            presets = sorted(file_times, key=lambda x: x[1], reverse=True)
            presets = [f[0] for f in presets]
            presets = preset_filter(presets)
            unique_presets = []
            for preset in presets:
                if preset not in unique_presets:
                    unique_presets.append(preset)
            presets = unique_presets
            if config.preset in presets:
                presets.remove(config.preset)
            presets.insert(0, config.preset)
            presets = presets[:shared.BUTTON_NUM]
            presets_list = ','.join(presets)

    if is_guest and hasattr(shared.token, 'set_local_vars_for_guest'):
        shared.token.set_local_vars_for_guest("user_presets", presets_list, user_session, ua_hash)
    else:

        shared.token.set_local_vars("user_presets", presets_list, user_session, ua_hash)
    return presets_list

preset_samples = {}
preset_samples_user_mtime = {}
preset_samples_base_mtime = {}
PRESET_COMPLETE_REFRESH_SECONDS = 8
preset_samples_complete_ts = {}
def get_preset_samples(user_did=None):
    global preset_samples, preset_samples_user_mtime, preset_samples_base_mtime, preset_samples_complete_ts
    cache_key = user_did if user_did else 'guest'
    path_preset = os.path.abspath(f'./presets/')
    base_files = [p for p in util.get_files_from_folder(path_preset, ['.json'], None) 
                 if is_preset_file_allowed(p)]
    base_presets = [p[:-5] for p in base_files]
    base_mtime = 0
    for f in base_files:
        try:
            base_mtime = max(base_mtime, os.path.getmtime(os.path.join(path_preset, f)))
        except Exception:
            continue

    user_path_preset = get_path_in_user_dir('presets', user_did) if user_did else get_path_in_user_dir('presets')
    user_presets = []
    user_mtime = 0
    if user_path_preset and os.path.exists(user_path_preset):
        presets2 = [p for p in util.get_files_from_folder(user_path_preset, ['.json'], None)
                   if is_preset_file_allowed(p)]
        for p in presets2:
            user_presets.append(f'{p[:-5]}.')
            try:
                user_mtime = max(user_mtime, os.path.getmtime(os.path.join(user_path_preset, p)))
            except Exception:
                continue

    store_list = _filter_existing_presets(PRESET_STORE_ORDER, user_did)

    if (
        cache_key in preset_samples
        and preset_samples_user_mtime.get(cache_key, -1) == user_mtime
        and preset_samples_base_mtime.get(cache_key, -1) == base_mtime
    ):
        cached = preset_samples[cache_key]
        now = time.time()
        last_ts = preset_samples_complete_ts.get(cache_key, 0)
        if now - last_ts < PRESET_COMPLETE_REFRESH_SECONDS:
            return cached
        refreshed = _apply_complete_markers(cached, user_did)
        preset_samples[cache_key] = refreshed
        preset_samples_complete_ts[cache_key] = now
        return refreshed

    if store_list:
        ordered_presets = store_list[:]
    else:
        ordered_presets = []
    existing = {_canonicalize_preset_name(p) for p in ordered_presets}
    # 追加：开发者硬编码之外的基础预置（按文件名字典序）
    for preset in sorted(base_presets):
        canonical = _canonicalize_preset_name(preset)
        if canonical not in existing:
            ordered_presets.append(preset)
            existing.add(canonical)
    # 追加：用户自定义预置（末尾，带.标识）
    for preset in user_presets:
        canonical = _canonicalize_preset_name(preset)
        if canonical not in existing:
            ordered_presets.append(preset)
            existing.add(canonical)

    refresh_model_list(ordered_presets, user_did)
    ordered_list = preset_filter([[p] for p in ordered_presets])
    marked_list = _apply_complete_markers(ordered_list, user_did)
    preset_samples[cache_key] = marked_list
    preset_samples_user_mtime[cache_key] = user_mtime
    preset_samples_base_mtime[cache_key] = base_mtime
    preset_samples_complete_ts[cache_key] = time.time()
    return marked_list


def refresh_preset_store_list(state):
    user_did = None
    try:
        if isinstance(state, dict) and 'user' in state and state['user']:
            user_did = state['user'].get_did()
    except Exception:
        user_did = None

    cache_key = user_did if user_did else 'guest'
    preset_samples.pop(cache_key, None)
    preset_samples_user_mtime.pop(cache_key, None)
    preset_samples_base_mtime.pop(cache_key, None)
    preset_samples_complete_ts.pop(cache_key, None)

    return gr.Dataset.update(samples=get_preset_samples(user_did))


def get_system_message():
    global config_ext

    fooocus_log = os.path.abspath(f'./update_log.md')
    simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
    update_msg_f = ''
    first_line_f = None
    if os.path.exists(fooocus_log):
        with open(fooocus_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_f is None:
                    first_line_f = line.strip()
                if line.strip() == config_ext['fooocus_line']:
                    break
                if first_line_f:
                    update_msg_f += line
                line = log_file.readline()
    update_msg_s = ''
    first_line_s = None
    if os.path.exists(simplesdxl_log):
        with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_s is None:
                    first_line_s = line.strip()
                if line.strip() == config_ext['simplesdxl_line']:
                    break
                if first_line_s:
                    update_msg_s += line
                line = log_file.readline()
    update_msg_f = update_msg_f.replace("\n","  ")
    update_msg_s = update_msg_s.replace("\n","  ")
    
    f_log_path = os.path.abspath("./update_log.md")
    s_log_path = os.path.abspath("./simplesdxl_log.md")
    if len(update_msg_f)>0:
        body_f = f'<b id="update_f">[Fooocus更新信息]</b>: {update_msg_f}<a href="{args_manager.args.webroot}/file={f_log_path}">更多>></a>   '
    else:
        body_f = '<b id="update_f"> </b>'
    if len(update_msg_s)>0:
        body_s = f'<b id="update_s">[系统消息 - 已更新内容]</b>: {update_msg_s}<a href="{args_manager.args.webroot}/file={s_log_path}">更多>></a>'
    else:
         body_s = '<b id="update_s"> </b>'
    import mistune
    body = mistune.html(body_f+body_s)
    if first_line_f and first_line_s and (first_line_f != config_ext['fooocus_line'] or first_line_s != config_ext['simplesdxl_line']):
        config_ext['fooocus_line']=first_line_f
        config_ext['simplesdxl_line']=first_line_s
        with open(enhanced_config, "w", encoding="utf-8") as config_file:
            json.dump(config_ext, config_file)
    return body if body else ''



def preset_instruction():
    head = "<div style='max-width:100%; max-height:86px; overflow:hidden'>"
    foot = "</div>"
    body = '预置包简介:<span style="position: absolute;right: 0;"><a href="https://gitee.com/metercai/SimpleSDXL/blob/SimpleSDXL/presets/readme.md">\U0001F4DD 什么是预置包</a></span>'
    body += f'<iframe id="instruction" src="{get_preset_inc_url()}" frameborder="0" scrolling="no" width="100%"></iframe>'
    
    return head + body + foot

get_system_params_js = '''
function(system_params) {
    const params = new URLSearchParams(window.location.search);
    const sessionCookie = getCookie('aitoken');
    const url_params = Object.fromEntries(params);
    if (url_params["__lang"]) 
        system_params["__lang"]=url_params["__lang"];
    if (url_params["__theme"]) 
        system_params["__theme"]=url_params["__theme"];
    if (sessionCookie) 
        system_params["__session"]=sessionCookie;
    return system_params;
}
'''

def init_nav_bars(state_params, comfyd_active_checkbox, fast_comfyd_checkbox, cache_clear_on_finish_checkbox, reserved_vram, cache_ram_enable, ram_pressure, minicpm_checkbox, minicpm_version, advanced_logs, wavespeed_strength, translation_methods, p2p_active_checkbox, p2p_remote_process, p2p_in_did_list, p2p_out_did_list, no_welcome_checkbox, missing_model_filter_checkbox, request: gr.Request):
    #logger.info(f'request.headers:{request.headers}')
    #logger.info(f'request.client:{request.client}')
    admin_currunt_value = [comfyd_active_checkbox, fast_comfyd_checkbox, cache_clear_on_finish_checkbox, reserved_vram, cache_ram_enable, ram_pressure, minicpm_checkbox, minicpm_version, advanced_logs, wavespeed_strength, translation_methods, p2p_active_checkbox, p2p_remote_process, p2p_in_did_list, p2p_out_did_list, no_welcome_checkbox, missing_model_filter_checkbox]
    #logger.info(f'admin_currunt_value: {admin_currunt_value}')

    user_agent = request.headers["user-agent"]
    ua_hash = hashlib.sha256(user_agent.encode('utf-8')).hexdigest()
    ua_session = gen_ua_session(request.client["host"], str(request.client["port"]), request.headers["user-agent"])
    state_params.update({"ua_hash": ua_hash})
    state_params.update({"ua_session": ua_session})
    if "__session" not in state_params.keys():
        sstoken = shared.token.get_guest_sstoken(ua_hash)
        state_params.update({"sstoken": sstoken})
        user_did = shared.token.get_guest_did()
        user_session = sstoken
        state_params.update({"__session": user_session})
        logger.info(f'New request/新请求(无身份): {request.client.host}:{request.client.port} --> {request.headers.host}, session({user_session})')
    else:
        #logger.info(f'aitoken: {state_params["__session"]}, guest={shared.token.get_guest_did()}')
        user_session = state_params["__session"]
        user_did = shared.token.check_sstoken_and_get_did(user_session, ua_hash)
        if user_did == "Unknown":
            sstoken = shared.token.get_guest_sstoken(ua_hash)
            state_params.update({"sstoken": sstoken})
            user_did = shared.token.get_guest_did()
            user_session = sstoken
            state_params.update({"__session": user_session})
            logger.debug(f'user-agent:{request.headers["user-agent"]}, cookie:{request.headers["cookie"]}')
            logger.info(f'Reset request/重置请求(无效身份): {request.client.host}:{request.client.port} --> {request.headers.host}, session({user_session})')
            user = shared.token.get_user_context(user_did)
        else:
            user = shared.token.get_user_context(user_did)
            state_params.update({"sstoken": ""})
            if user.get_nickname().startswith('guest_'):
                logger.info(f'Reset request/游客请求: {request.client.host}:{request.client.port} --> {request.headers.host}, session({user_session})')
            else:
                logger.info(f'Binded request/含身份请求: {request.client.host}:{request.client.port} --> {request.headers.host}, session({user_session})')
    shared.token.log_register(state_params["__session"])
    state_params.update({"user": shared.token.get_user_context(user_did)})
    state_params.update({"sys_did":  shared.token.get_sys_did()})
    state_params.update({"local_access":  True if request.client.host == shared.args.listen or shared.args.listen=='127.0.0.1' or request.client.host=='127.0.0.1' else False})

    if "__lang" not in state_params.keys():
        if 'accept-language' in request.headers and 'zh-CN' in request.headers['accept-language']:
            args_manager.args.language = 'cn'
        state_params.update({"__lang": ads.get_user_default("__lang", state_params, args_manager.args.language)})
    if "__theme" not in state_params.keys():
        state_params.update({"__theme": ads.get_user_default("__theme", state_params, args_manager.args.theme)})
    if "__preset" not in state_params.keys():
        state_params.update({"__preset": config.preset})
    if "__is_mobile" not in state_params.keys():
        state_params.update({"__is_mobile": False if user_agent.find("Mobile")>0 and user_agent.find("AppleWebKit")>0 else False})
    if "__webpath" not in state_params.keys():
        state_params.update({"__webpath": f'{args_manager.args.webroot}/file={os.getcwd()}'})
    if "__max_per_page" not in state_params.keys():
        if state_params["__is_mobile"]:
            state_params.update({"__max_per_page": 9})
        else:
            state_params.update({"__max_per_page": 18})
    if "__max_catalog" not in state_params.keys():
        state_params.update({"__max_catalog": config.default_image_catalog_max_number })
    state_params.update({"infobox_state": 0})
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"array_wildcards_mode": '_'})
    state_params.update({"wildcard_in_wildcards": 'root'})
    state_params.update({"bar_button": config.preset})
    state_params.update({"preset_store": False})
    state_params.update({"engine": 'Z-image'})
    state_params.update({"engine_type": 'image'})
    results = [gr.update(value=get_welcome_image(config.preset,state_params["__is_mobile"],no_welcome=ads.get_admin_default("no_welcome_checkbox")))]
    results += [gr.update(value=modules.flags.language_radio(state_params["__lang"])), gr.update(value=state_params["__theme"])]
    preset = 'Z-imageT'
    preset_url = get_preset_inc_url(preset)
    state_params.update({"__preset_url":preset_url})
    results += [gr.update(visible=True if 'blank.inc.html' not in preset_url else False)]
    results += get_all_user_default(state_params)
    results += get_all_admin_default(admin_currunt_value)

    return results

def get_preset_inc_url(preset_name='blank'):
    preset_name = f'{preset_name}.inc'
    preset_inc_path = os.path.abspath(f'./presets/html/{preset_name}.html')
    blank_inc_path = os.path.abspath(f'./presets/html/blank.inc.html')
    if os.path.exists(preset_inc_path):
        return f'{args_manager.args.webroot}/file={preset_inc_path}'
    else:
        return f'{args_manager.args.webroot}/file={blank_inc_path}'

def refresh_nav_bars(state_params):
    # 安全获取__session和ua_hash，如果不存在则提供默认值
    user_session = state_params.get("__session", "")
    ua_hash = state_params.get("ua_hash", "")
    preset_name_list = get_preset_name_list(user_session, ua_hash).split(',')
    filtered_presets = preset_filter(preset_name_list)
    preset_name_list = [preset for preset in preset_name_list if preset in filtered_presets]

    unique_presets = []
    for preset in preset_name_list:
        if preset and preset not in unique_presets:
            unique_presets.append(preset)
    preset_name_list = unique_presets
    user_did = state_params["user"].get_did()
    is_guest = shared.token.is_guest(user_did)
    path_preset = os.path.abspath(f'./presets/')
    user_path_preset = get_path_in_user_dir('presets', user_did)
    num = len(preset_name_list)
    for preset in preset_name_list[:]:
        arch_str = config.get_gpu_arch_str_in_preset_name()
        if preset.endswith('.'):
            preset_file = os.path.join(user_path_preset, f'{preset[:-1]}.json')
            preset_file2 = os.path.join(user_path_preset, f'{preset[:-1]}{arch_str}.json')
        else:
            preset_file = os.path.join(path_preset, f'{preset}.json')
            preset_file2 = os.path.join(path_preset, f'{preset}{arch_str}.json')
        if os.path.exists(preset_file2):
            preset_file = preset_file2
        if not os.path.exists(preset_file):
            preset_name_list.remove(preset)
    if num != len(preset_name_list):
        nav_name_list = ','.join(preset_name_list)
        if is_guest and hasattr(shared.token, 'set_local_vars_for_guest'):
            shared.token.set_local_vars_for_guest("user_presets", nav_name_list, state_params["__session"], state_params["ua_hash"])
        else:
            shared.token.set_local_vars("user_presets", nav_name_list, state_params["__session"], state_params["ua_hash"])

    for i in range(shared.BUTTON_NUM - len(preset_name_list)):
        preset_name_list.append('')
    results = []
    if state_params["__is_mobile"]:
        results += [gr.update(visible=False)]
    else:
        results += [gr.update(visible=True)]
    for i in range(len(preset_name_list)):
        name = preset_name_list[i]
        name += '\u2B07' if is_models_file_absent(name, user_did) else ''
        visible_flag = i < (7 if state_params["__is_mobile"] else shared.BUTTON_NUM)
        if name:
            results += [gr.update(value=name, interactive=True, visible=visible_flag)]
        else: 
            results += [gr.update(value='', interactive=False, visible=False)]
    return results
def wait_for_minicpm_completion(check_interval=0.5):
    try:
        while True:
            processing_status = MiniCPM.get_processing_status()
            is_processing = False
            if isinstance(processing_status, bool):
                is_processing = processing_status

            if not is_processing:
                return True
            time.sleep(check_interval)

    except Exception as e:
        logger.error(f"MiniCPM Error: {str(e)}")
        return True
def avoid_empty_prompt_for_scene(prompt, state, canvas_image, input_image1, scene_theme, additional_prompt, additional_prompt_2):
    describe_prompt = None
    if not prompt and 'scene_frontend' in state:
        visible = state["scene_frontend"].get('disvisible', [])
        canvas_visible = 'scene_canvas_image' not in visible
        canvas_img = meta_parser.extract_scene_image(canvas_image) if canvas_visible else None
        input_img = meta_parser.extract_scene_image(input_image1)
        use_img = canvas_img if canvas_img is not None else input_img
        describe_prompt, img_is_ok = describe_prompt_for_scene(state, use_img, scene_theme, f'{additional_prompt}{additional_prompt_2}')
    return gr.update() if describe_prompt is None else describe_prompt


def process_before_generation(state_params, seed_random, image_seed, backend_params, scene_theme, scene_canvas_image, scene_input_image1, scene_input_image2, scene_additional_prompt, scene_additional_prompt_2, scene_var_number, scene_var_number2, scene_var_number3, scene_var_number4, scene_var_number5, scene_var_number6, scene_var_number7, scene_var_number8, scene_var_number9, scene_var_number10, scene_steps, scene_switch_option1, scene_switch_option2, scene_switch_option3, scene_switch_option4, scene_aspect_ratio, scene_image_number, scene_video, scene_audio, scene_original_video_path, active_video_source, sam3_input_video, sam3_original_video_path, sam3_mask_video):
    backend_params.update(dict(
        nickname=state_params["user"].get_nickname(),
        user_did=state_params["user"].get_did(),
        preset=state_params["__preset"],
        engine_type=state_params.get("engine_type", "image"),
        ))

    if scene_audio is not None and not (isinstance(scene_audio, str) and os.path.exists(scene_audio)):
        try:
            from extras.media_normalize import normalize_gradio_audio_value
            scene_audio = normalize_gradio_audio_value(scene_audio)
        except Exception:
            pass

    user_did = state_params["user"].get_did()
    if shared.token.is_admin(user_did):
        admin_outputs = os.path.abspath(os.path.join(shared.token.get_path_in_user_dir(user_did, "outputs"), 'ComfyUI'))
        if not os.path.exists(admin_outputs):
            os.makedirs(admin_outputs)
        
        # Update runtime
        if comfyd.is_running():
            comfyd.modify_variable({"outputs": admin_outputs})
            
        # Update startup args for persistence
        try:
            for arg in comfyd.comfyd_args:
                if len(arg) >= 2 and arg[0] == "--output-directory":
                    arg[1] = admin_outputs
                    # print(f"Updated Comfyd startup args output directory to: {admin_outputs}")
                    break
        except Exception as e:
            print(f"Error updating comfyd startup args: {e}")
    
    if 'scene_frontend' in state_params:
        scene_frontend = state_params['scene_frontend']
        disvisible = scene_frontend.get('disvisible', []) if isinstance(scene_frontend, dict) else []
        if not isinstance(disvisible, list):
            disvisible = []
        disvisible = set(disvisible)

        if 'scene_canvas_image' in disvisible:
            scene_canvas_image = None
        if 'scene_input_image1' in disvisible:
            scene_input_image1 = None
        if 'scene_input_image2' in disvisible:
            scene_input_image2 = None
        if 'scene_video' in disvisible:
            scene_video = None
        if 'scene_audio' in disvisible:
            scene_audio = None
        if 'scene_var_number' in disvisible:
            scene_var_number = None
        if 'scene_var_number2' in disvisible:
            scene_var_number2 = None
        if 'scene_var_number3' in disvisible:
            scene_var_number3 = None
        if 'scene_var_number4' in disvisible:
            scene_var_number4 = None
        if 'scene_var_number5' in disvisible:
            scene_var_number5 = None
        if 'scene_var_number6' in disvisible:
            scene_var_number6 = None
        if 'scene_var_number7' in disvisible:
            scene_var_number7 = None
        if 'scene_var_number8' in disvisible:
            scene_var_number8 = None
        if 'scene_var_number9' in disvisible:
            scene_var_number9 = None
        if 'scene_var_number10' in disvisible:
            scene_var_number10 = None
        if 'scene_switch_option1' in disvisible:
            scene_switch_option1 = None
        if 'scene_switch_option2' in disvisible:
            scene_switch_option2 = None
        if 'scene_switch_option3' in disvisible:
            scene_switch_option3 = None
        if 'scene_switch_option4' in disvisible:
            scene_switch_option4 = None

        scene_additional_prompt = f'{scene_additional_prompt}{scene_additional_prompt_2}'
        if util.is_chinese(scene_additional_prompt) and not scene_frontend['task_method'][scene_theme].lower().endswith('_cn'):
            scene_additional_prompt = minicpm.translate(scene_additional_prompt, 'Slim Model')
        resize_image_flag = True
        mask_color_flag = False
        preprocessor_methods = modules.flags.get_value_by_scene_theme(state_params, scene_theme, 'image_preprocessor_method', [])
        if len(preprocessor_methods)>0:
            for preprocessor_method in preprocessor_methods:
                if '-normalization' in preprocessor_method:
                    resize_image_flag = False
                if 'mask_color' in preprocessor_method:
                    mask_color_flag = True
        if scene_input_image1 is not None:
            scene_input_image1 = util.resize_image(scene_input_image1, max_side=1280, resize_mode=4) if resize_image_flag else scene_input_image1
        if scene_input_image2 is not None:
            scene_input_image2 = util.resize_image(scene_input_image2, max_side=1280, resize_mode=4) if resize_image_flag else scene_input_image2

        if scene_canvas_image is not None:
            image = scene_canvas_image['image']
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
            white_background = np.full_like(rgb, 255, dtype=np.uint8)
            mask = alpha > 0
            image = np.where(np.expand_dims(mask, axis=-1), rgb, white_background)
            image = np.dstack((image, alpha))
            scene_canvas_image['image'] = util.resize_image(image, max_side=1280, resize_mode=4) if resize_image_flag else image
            mask = scene_canvas_image['mask']
            if mask.shape[2] == 4:
                if mask_color_flag:
                    color = mask[:, :, 0:3].astype(np.float32)
                    alpha = mask[:, :, 3:4].astype(np.float32) / 255.0
                    mask = color * alpha
                    mask = mask.clip(0, 255).astype(np.uint8)
                else:
                    alpha = mask[:, :, 3]
                    h, w = alpha.shape
                    mask = np.zeros((h, w, 3), dtype=np.uint8)
                    mask[:, :, 0] = alpha
                    mask[:, :, 1] = alpha
                    mask[:, :, 2] = alpha
            scene_canvas_image['mask'] = util.resize_image(util.HWC3(mask), max_side=1280, resize_mode=4) if resize_image_flag else mask

        scene_video_effective = scene_original_video_path if scene_original_video_path else scene_video
        sam3_video_effective = sam3_original_video_path if sam3_original_video_path else sam3_input_video
        if active_video_source == "scene":
            video_effective = scene_video_effective if scene_video_effective else sam3_video_effective
        elif active_video_source == "sam3":
            video_effective = sam3_video_effective if sam3_video_effective else scene_video_effective
        else:
            video_effective = sam3_video_effective if sam3_video_effective else scene_video_effective
        backend_params.update(dict(
            task_method=f'scene_{scene_frontend["task_method"][scene_theme]}',
            scene_frontend=scene_frontend['version'],
            scene_canvas_image=scene_canvas_image,
            scene_input_image1=scene_input_image1,
            scene_input_image2=scene_input_image2,
            scene_theme=scene_theme,
            scene_additional_prompt=scene_additional_prompt,
            scene_var_number=None if 'var_number' not in scene_frontend else scene_var_number,
            scene_var_number2=scene_var_number2,
            scene_var_number3=scene_var_number3,
            scene_var_number4=scene_var_number4,
            scene_var_number5=scene_var_number5,
            scene_var_number6=scene_var_number6,
            scene_var_number7=scene_var_number7,
            scene_var_number8=scene_var_number8,
            scene_var_number9=scene_var_number9,
            scene_var_number10=scene_var_number10,
            scene_switch_option1=scene_switch_option1,
            scene_switch_option2=scene_switch_option2,
            scene_switch_option3=scene_switch_option3,
            scene_switch_option4=scene_switch_option4,
            scene_aspect_ratio=scene_aspect_ratio.split('|')[0] if '×' in scene_aspect_ratio else modules.flags.scene_aspect_ratios_size[scene_aspect_ratio],
            scene_image_number=scene_image_number,
            video=video_effective,
            audio=scene_audio,
            mask_video=sam3_mask_video,
            scene_steps=scene_steps if 'scene_steps' in scene_frontend else None
            ))
    state_params["absent_model"] = False
    if not args_manager.args.disable_backend and is_models_file_absent(state_params["__preset"], state_params["user"].get_did()):
        if not ads.get_user_default("no_model_modal_checkbox", state_params, False):
            gr.Info(preset_absent_model_note_info)
        state_params["absent_model"] = True
        # if shared.token.is_admin(state_params["user"].get_did()):
        #     download_model_files(state_params["__preset"], state_params["user"].get_did(), True)

    superprompter.remove_superprompt()
    remove_tokenizer()
    minicpm.free_model()
    try:
        import extras.wd14tagger
        extras.wd14tagger.free_model()
    except Exception:
        pass

    # stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box
    results = [gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False), gr.update(visible=False, interactive=False), [], True, gr.update(visible=False, open=False), gr.update(visible=False), gr.update(visible=False)]
    # image_seed
    if seed_random:
        seed_value = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(image_seed)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                 pass
        except ValueError:
            seed_value = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    results += [seed_value]
    # random_button, super_prompter, background_theme, image_tools_checkbox, bar_store_button, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(get_preset_name_list(state_params["__session"], state_params["ua_hash"]).split(','))
    results += [gr.update(interactive=False)] * (preset_nums + 5)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    # preset_store, identity_dialog
    results += [gr.update(visible=False)]*2

    state_params["gallery_state"]='preview'
    state_params["preset_store"]=False
    state_params["identity_dialog"]=False
    return results


def process_after_generation(state_params):
    if "__max_per_page" not in state_params.keys():
        state_params.update({"__max_per_page": 18})
    if "__max_catalog" not in state_params.keys():
        state_params.update({"__max_catalog": config.default_image_catalog_max_number })
    
    max_per_page = state_params["__max_per_page"]
    max_catalog = state_params["__max_catalog"]
    user_did = state_params["user"].get_did()
    engine_type = state_params["engine_type"]
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog, user_did, engine_type)
    state_params.update({"__output_list": output_list})
    state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
    # generate_button, stop_button, skip_button, state_is_generating
    results = [gr.update(visible=True, interactive=True)] + [gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False]
    # gallery_index, index_radio
    results += [gr.update(choices=state_params["__output_list"], value=None), gr.update(visible=len(state_params["__output_list"])>0, open=False)]
    # random_button, super_prompter, background_theme, image_tools_checkbox, bar_store_button, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(get_preset_name_list(state_params["__session"], state_params["ua_hash"]).split(','))
    results += [gr.update(interactive=True)] * (preset_nums + 5)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    # [history_link, gallery_index_stat]
    results += [state_params['__finished_nums_pages']]
    results += [update_history_link(user_did, state_params["local_access"])]
    

    if len(state_params["__output_list"]) > 0 and engine_type == 'image':
        try:
            output_index = state_params["__output_list"][0].split('/')[0]
            gallery_util.refresh_images_catalog(output_index, True, user_did)
        except Exception as e:
            logger.error(f'Error in post-generation gallery processing: {e}')
   
    return results


def sync_message(state_params):
    state_params.update({"__message":system_message})
    return

preset_down_note_info = 'The model file is missing. You can click to download the required models. You can also use the model_checker to complete the model files.'
preset_downing_note_info = 'Downloading the model file required for image generation, please wait for a moment...'
preset_absent_model_note_info = 'The preset package being loaded has model files that need to be downloaded.'

def check_absent_model(bar_button, state_params):
    #logger.info(f'check_absent_model,state_params:{state_params}')
    state_params.update({'bar_button': bar_button})
    return 

def down_absent_model(state_params):
    state_params.update({'bar_button': state_params["bar_button"].replace('\u2B07', '')})
    return gr.update(visible=False), state_params

reset_layout_num = 0
reset_layout_ui_outputs_len = 0

def reset_layout_ui(prompt, negative_prompt, state_params, is_generating, inpaint_mode, comfyd_active_checkbox, bar_button = None):
    global system_message, preset_down_note_info, reset_layout_num, reset_layout_ui_outputs_len

    if bar_button is not None:
        state_params.update({"bar_button": bar_button})

    if "__lang" not in state_params:
        state_params["__lang"] = ads.get_user_default("__lang", state_params, args_manager.args.language)
    if "__theme" not in state_params:
        state_params["__theme"] = ads.get_user_default("__theme", state_params, args_manager.args.theme)

    state_params.update({"__message": system_message})
    system_message = 'system message was displayed!'
    if '__preset' not in state_params.keys() or 'bar_button' not in state_params.keys() or state_params["__preset"]==state_params['bar_button']:
        # Default reset for comparison UI when not switching
        comparison_default = [gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]
        nav_updates = refresh_nav_bars(state_params)
        fill_count = reset_layout_ui_outputs_len - len(nav_updates)
        if fill_count < 0:
            fill_count = 0
        return nav_updates + [gr.update()] * fill_count + [state_params] + comparison_default
    preset = state_params["bar_button"] if '\u2B07' not in state_params["bar_button"] else state_params["bar_button"].replace('\u2B07', '')
    logger.info(f'Reset_context: preset={state_params.get("__preset", None)}-->{preset}, theme={state_params.get("__theme", None)}, lang={state_params.get("__lang", None)}')
    if not args_manager.args.disable_backend and '\u2B07' in state_params["bar_button"]:
        if not ads.get_user_default("no_model_modal_checkbox", state_params, False):
            gr.Info(preset_down_note_info)

    state_params.update({"__preset": preset})

    comparison_outputs = [False, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=get_welcome_image(preset, state_params["__is_mobile"], no_welcome=ads.get_admin_default("no_welcome_checkbox")))]

    config_preset = config.try_get_preset_content(preset, state_params["user"].get_did())
    preset_prepared = meta_parser.parse_meta_from_preset(config_preset)

    engine = preset_prepared.get('engine', {}).get('backend_engine', 'Fooocus')
    engine_type = preset_prepared.get('engine', {}).get('engine_type', 'image')
    state_params.update({"engine": engine})
    state_params.update({"engine_type": engine_type})
    scene_frontend = preset_prepared.get('engine', {}).get('scene_frontend', None)
    if scene_frontend:
        state_params.update({"scene_frontend": scene_frontend})
        task_method = scene_frontend['task_method']
        if isinstance(task_method, list):
            task_method = task_method[0]
        elif isinstance(task_method, dict):
            if task_method:
                task_method = task_method[next(iter(task_method))]
    else:
        if 'scene_frontend' in state_params:
            del state_params["scene_frontend"]
        task_method = preset_prepared.get('engine', {}).get('backend_params', modules.flags.get_engine_default_backend_params(engine)).get('task_method', 'text2image')
        if isinstance(task_method, str) and engine == 'Fooocus':
            task_method = 'text2image'
    state_params.update({"task_method": task_method})
    preset_prepared.update({
        'preset': preset,
        'task_method': task_method,
        'is_mobile': state_params["__is_mobile"] })

    state_params['__preset_prepared'] = preset_prepared # Cache for reset_layout_values
    preset_url = preset_prepared.get('reference', get_preset_inc_url(preset))
    state_params.update({"__preset_url":preset_url})
    state_params.update({'preset_store': False})

    results = refresh_nav_bars(state_params)
    results += meta_parser.switch_layout_template(preset_prepared, state_params, preset_url)
    fill_count = reset_layout_ui_outputs_len - len(results)
    if fill_count > 0:
        results += [gr.update()] * fill_count
    elif fill_count < 0:
        results = results[:reset_layout_ui_outputs_len]

    # comparison_state, comparison_box, progress_gallery, compare_btn, progress_window
    comparison_outputs = [False, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=get_welcome_image(preset, state_params["__is_mobile"], no_welcome=ads.get_admin_default("no_welcome_checkbox")))]
    
    return results + [state_params] + comparison_outputs

def reset_layout_values(state_params, is_generating, inpaint_mode, use_resolution_override, scene_batch_target, scene_theme=None, scene_aspect_ratio=None):
    if not isinstance(state_params, dict):
        state_params = {}

    preset = state_params.get("__preset", None)
    if not preset:
        preset = config.preset
        state_params["__preset"] = preset

    preset_prepared = state_params.get('__preset_prepared', None)
    if preset_prepared is None:
        try:
            user = state_params.get("user", None)
            user_did = user.get_did() if user is not None and hasattr(user, "get_did") else None
        except Exception:
            user_did = None
        if not user_did:
            try:
                user_did = shared.token.get_guest_did()
            except Exception:
                user_did = None

        config_preset = config.try_get_preset_content(preset, user_did)
        preset_prepared = meta_parser.parse_meta_from_preset(config_preset)

        engine = preset_prepared.get('engine', {}).get('backend_engine', 'Fooocus')
        state_params.update({"engine": engine})
        state_params.update({"backend_engine": engine})
        task_method = preset_prepared.get('engine', {}).get('backend_params', modules.flags.get_engine_default_backend_params(engine)).get('task_method', 'text2image')
        if isinstance(task_method, str) and engine == 'Fooocus' and task_method.startswith('z_image_'):
            task_method = 'text2image'
        state_params.update({"task_method": task_method})
        preset_prepared.update({
            'preset': preset,
            'task_method': task_method,
            'is_mobile': state_params.get("__is_mobile", False) })

    results = meta_parser.load_parameter_button_click(preset_prepared, is_generating, inpaint_mode, use_resolution_override, no_welcome=ads.get_admin_default("no_welcome_checkbox"))

    def _parse_scene_resolution(value):
        if value is None:
            return -1, -1
        s = str(value).strip()
        if not s:
            return -1, -1
        s = s.split(",", 1)[0].strip()
        if "|" in s:
            parts = s.split("|", 1)
            w_raw = parts[0].strip()
            ratio = parts[1].strip()
            if w_raw.isdigit() and ":" in ratio:
                try:
                    a_str, b_str = ratio.split(":", 1)
                    a = float(a_str)
                    b = float(b_str)
                    w = int(w_raw)
                    if a > 0 and b > 0 and w > 0:
                        h = int(round(w * (b / a)))
                        return w, h
                except Exception:
                    pass
        try:
            import re
            m = re.search(r"(\d+)\D+(\d+)", s.replace("×", "x").replace("*", "x"))
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
        return -1, -1

    def _scene_is_t2v(state):
        try:
            scenes = state.get("scene_frontend", {})
            if not isinstance(scenes, dict):
                return False
            task_method = scenes.get("task_method", "")
            theme = scene_theme if isinstance(scene_theme, str) and scene_theme else state.get("scene_theme", None)
            if isinstance(task_method, dict):
                if isinstance(theme, str) and theme in task_method:
                    task_method = task_method.get(theme, "")
                elif task_method:
                    task_method = next(iter(task_method.values()), "")
            elif isinstance(task_method, list):
                task_method = task_method[0] if task_method else ""
            return "t2v" in str(task_method or "").lower()
        except Exception:
            return False

    scene_t2v_enabled = _scene_is_t2v(state_params)
    if scene_t2v_enabled:
        try:
            ow = results[13]
            oh = results[14]
            ow_val = int(ow) if isinstance(ow, (int, float)) else -1
            oh_val = int(oh) if isinstance(oh, (int, float)) else -1
        except Exception:
            ow_val, oh_val = -1, -1

        if not (ow_val > 0 and oh_val > 0):
            w2, h2 = _parse_scene_resolution(scene_aspect_ratio)
            if w2 > 0 and h2 > 0:
                results[13] = gr.update(value=int(w2))
                results[14] = gr.update(value=int(h2))
    results += update_after_identity_sub(state_params)

    reset_ui_results = [gr.update(), gr.update(), gr.update()] + \
               [gr.update()]*4 + [True] + [False] + \
               [gr.update(visible=False),False,[],"base",gr.update(variant="secondary"),gr.update(variant="secondary")] + \
               [gr.update(visible=False) for _ in config.default_loras] + \
               [False for _ in config.default_loras] + \
               [[] for _ in config.default_loras] + \
               [gr.update(variant="secondary") for _ in config.default_loras]
    results += reset_ui_results

    sync_intput_reserved()
    ldm_patched.modules.model_management.print_memory_info("after switched preset")

    try:
        import modules.batch_utils as batch_utils
        batch_accordion_update = batch_utils.refresh_scene_batch_accordion(state_params)
        batch_target_update = batch_utils.refresh_scene_batch_target(state_params, scene_batch_target)
    except Exception:
        batch_accordion_update = gr.update()
        batch_target_update = gr.update()

    return [batch_accordion_update, batch_target_update] + results

def check_admin_exists():
    try:
        return bool(shared.token.get_admin_did())
    except Exception as e:
        print(f"[DEBUG] 获取管理员变量时出错: {str(e)}")
        return False

def toggle_preset_store(state):
    user_in_state = 'user' in state
    store_update = gr.update(samples=get_preset_samples(state["user"].get_did() if user_in_state else None))
    is_guest = shared.token.is_guest(state["user"].get_did()) if user_in_state else True
    if user_in_state and not is_guest:
        if 'preset_store' in state:
            flag = state['preset_store']
        else:
            state['preset_store'] = False
            flag = False
        state['preset_store'] = not flag
        state['identity_dialog'] = False
        return [gr.update(visible=not flag), store_update] + update_topbar_js_params(state) + [gr.update(visible=False)] + [gr.update()]*17
    else:
        has_admin = False
        has_admin = check_admin_exists()

        if not has_admin:
            if 'preset_store' in state:
                flag = state['preset_store']
            else:
                state['preset_store'] = False
                flag = False
            state['preset_store'] = not flag
            state['identity_dialog'] = False
            return [gr.update(visible=not flag), store_update] + update_topbar_js_params(state) + [gr.update(visible=False)] + [gr.update()]*17
        else:
            return [gr.update(), store_update] + update_topbar_js_params(state) + toggle_identity_dialog(state)

def update_navbar_from_mystore(selected_preset, state):
    global preset_samples
    user_did = state["user"].get_did()
    is_guest = shared.token.is_guest(user_did)

    def _to_index(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            if not v:
                return None
            v = v[0]
        try:
            return int(v)
        except Exception:
            return None

    idx = _to_index(selected_preset)
    samples = get_preset_samples(user_did)
    if idx is None or idx < 0 or idx >= len(samples):
        if is_guest:
            samples = get_preset_samples(None)
        if idx is None or idx < 0 or idx >= len(samples):
            return refresh_nav_bars(state) + update_topbar_js_params(state)

    selected_preset_name = samples[idx][0]
    selected_preset_name = _strip_preset_marker(selected_preset_name)

    results = refresh_nav_bars(state)
    results2 = update_topbar_js_params(state)
    nav_name_list = get_preset_name_list(state["__session"], state["ua_hash"])
    nav_array = nav_name_list.split(',')
    if 'Z-imageT' not in nav_array:
        nav_array.insert(0, 'Z-imageT')
    else:
        nav_array.remove('Z-imageT')
        nav_array.insert(0, 'Z-imageT')
    available_presets_count = 0

    missing_model_filter = ads.get_admin_default("missing_model_filter_checkbox")

    filtered_nav_array = []
    for preset in nav_array:
        if preset:
            if not missing_model_filter or not is_models_file_absent(preset, user_did):
                available_presets_count += 1
                filtered_nav_array.append(preset)
            else:
                logger.info(f'[Preset Management] Filtered out preset with missing model: {preset}')

    if len(filtered_nav_array) != len([p for p in nav_array if p]):
        nav_array = filtered_nav_array
        if 'user' in state and not is_guest:
            filtered_nav_name_list = ','.join(nav_array)
            shared.token.set_local_vars("user_presets", filtered_nav_name_list, state["__session"], state["ua_hash"])

    if selected_preset_name in ["Z-imageT", state["__preset"]]:
        return results + results2
    if selected_preset_name in nav_array:
        nav_array.remove(selected_preset_name)
        logger.info(f'[Preset Management] Withdraw the preset/回撤预置包: {selected_preset_name}.')
    else:
        elimination_threshold = max(available_presets_count, len(nav_array))
        if elimination_threshold >= shared.BUTTON_NUM:
            if state["__preset"] not in nav_array:
                return results + results2
            position = nav_array.index(state["__preset"])
            if position+1 == shared.BUTTON_NUM:
                nav_array = nav_array[:-2] + nav_array[-1:]
            else:
                nav_array = nav_array[:-1]
        nav_array.append(selected_preset_name)
        logger.info(f'[Preset Management] Launch the preset/启用预置包: {selected_preset_name}.')

    nav_name_list = ','.join(nav_array)
    if 'user' in state:
        if not is_guest:
            logger.info(f"[Preset Management] save mypreset: {nav_name_list}")
            shared.token.set_local_vars("user_presets", nav_name_list, state["__session"], state["ua_hash"])
        else:
            has_admin = False
            has_admin = check_admin_exists()
            
            if not has_admin:
                shared.token.set_local_vars("user_presets", nav_name_list, state["__session"], state["ua_hash"])

    try:
        return refresh_nav_bars(state) + update_topbar_js_params(state)
    except TypeError as e:
        logger.error(f"UI Update Error: {str(e)}")

def admin_sync_to_guest(state, catalog='presets'):
    user_did = state["user"].get_did()
    if shared.token.is_admin(user_did):
        if catalog == 'presets':
            nav_name_list = get_preset_name_list(state["__session"], state["ua_hash"])
            shared.token.set_local_vars_for_guest("user_presets", nav_name_list, state["__session"], state["ua_hash"])
    current_time = datetime.now().strftime("%H:%M:%S")
    admin_sync_title = 'Sync presets nav to guest' if state["__lang"]!='cn' else '同步预置导航给游客'
    logger.info(f'Sync presets nav to guest: {current_time}')
    return f'{admin_sync_title}({current_time})'



def update_topbar_js_params(state):
    # 获取原始预设列表
    nav_name_list = get_preset_name_list(state["__session"], state["ua_hash"])
    # 对预设列表进行过滤，确保与UI显示的预设一致
    preset_name_list = nav_name_list.split(',')
    filtered_presets = preset_filter(preset_name_list)
    # 创建一个仅包含过滤后预设的新列表
    filtered_preset_name_list = [preset for preset in preset_name_list if preset in filtered_presets]
    filtered_nav_name_list_str = ','.join(filtered_preset_name_list)

    if "__lang" not in state:
        state["__lang"] = ads.get_user_default("__lang", state, args_manager.args.language)
    if "__theme" not in state:
        state["__theme"] = ads.get_user_default("__theme", state, args_manager.args.theme)
    if "__preset" not in state:
        state["__preset"] = config.preset

    system_params= dict(
        __preset=state.get("__preset"),
        __theme=state.get("__theme"),
        __nav_name_list=filtered_nav_name_list_str,  # 使用过滤后的预设列表
        sstoken=state["sstoken"],
        user_name=state["user"].get_nickname(),
        user_did=state["user"].get_did(),
        user_role='guest' if shared.token.is_guest(state["user"].get_did()) else 'admin' if shared.token.is_admin(state["user"].get_did()) else 'member',
        upstream=shared.upstream_did,
        task_class_name=state["engine"],
        preset_store=state["preset_store"],
        __message='' if "__message" not in state else state["__message"],
        __webpath=state["__webpath"],
        __lang=state.get("__lang"),
        __preset_url=state["__preset_url"],
        __finished_nums_pages=state["__finished_nums_pages"],
        user_qr="" if 'user_qr' not in state else state.pop("user_qr"),
        engine_type=state['engine_type'],
        no_welcome_image=ads.get_admin_default("no_welcome_checkbox"),
        missing_model_filter=ads.get_admin_default("missing_model_filter_checkbox")
        )
    return [system_params]


def export_identity(state):
    if not shared.token.is_guest(state["user"].get_did()):
        state["user_qr"] = export_identity_qrcode_svg(state["user"].get_did())
        #logger.info(f'user_qrcode_svg: {state["user_qr"]}')
    elif shared.token.get_node_mode()!='online':
        admin_qr = shared.token.export_isolated_admin_qrcode_svg()
        if admin_qr:
            state["user_qr"] = admin_qr
            #logger.info(f'admin_user_qrcode_svg: {state["user_qr"]}')
    return update_topbar_js_params(state)[0]


def update_history_link(user_did, local_access):
    log_link = '' if args_manager.args.disable_image_log else f'<a href="file={get_current_html_path(None, user_did)}" target="_blank">\U0001F4DA History Log</a>'
    return gr.update(value=log_link) 

def update_comfyd_url(state):
    entry_url = f'http://{args_manager.args.listen}:{shared.sysinfo["loopback_port"]}{args_manager.args.webroot}/'
    user_did = state["user"].get_did() if "user" in state and state["user"] else None

    entry_point = ""
    entry_point_id = comfyd.get_entry_point_id()
    if entry_point_id is not None and user_did:
        try:
            entry_point = shared.token.get_entry_point(user_did, entry_point_id) or ""
            if not entry_point and hasattr(shared.token, "get_admin_did"):
                admin_did = shared.token.get_admin_did()
                if admin_did:
                    entry_point = shared.token.get_entry_point(admin_did, entry_point_id) or ""
            if not entry_point:
                sys_did = state.get("sys_did") if isinstance(state, dict) else None
                if sys_did and sys_did != user_did:
                    entry_point = shared.token.get_entry_point(sys_did, entry_point_id) or ""
        except Exception:
            entry_point = ""

    if not entry_point:
        hour_key = datetime.now().strftime("%Y%m%d%H")
        suffix = state.get("__session") or state.get("ua_hash") or (user_did or "guest")
        entry_point = f"{hour_key}_{suffix}"

    return f'<a href="{entry_url}?p={entry_point}" target="_blank">{entry_url}</a><div>Click and Entry embedded ComfyUI from here.</div>'
   
identity_introduce = '''
<div style="line-height:1.55">
  <div style="font-weight:600;margin-bottom:6px;">当前为游客</div>
  <div style="margin-bottom:10px;">点击“身份管理”绑定身份，可解锁更多功能。</div>
  <details style="margin:6px 0;">
    <summary style="cursor:pointer;">绑定身份后能做什么</summary>
    <div style="margin-top:6px;">
      1，为本机启用多用户模式，“预置包”变更为“我的预置”，私有化排列<br>
      2，独立的出图存储空间和日志历史页，保障隐私安全。<br>
      3，可将当前环境参数保存为个人定制的预置包。<br>
      4，解锁更多的功能配置管理和个性化服务。<br>
    </div>
  </details>
  <details style="margin:6px 0;">
    <summary style="cursor:pointer;">管理员与协助</summary>
    <div style="margin-top:6px;">
      系统指定首个绑定身份者为管理员，赋予超级管理权限：<br>
      1，可管理游客的预置导航及下载预置包所需模型。<br>
      2，管理高级系统设置，最高权限分配系统资源。<br>
      更多管理需求可以入QQ群:1005085136 进行交流。<br>
    </div>
  </details>
  <details style="margin:6px 0;">
    <summary style="cursor:pointer;">身份机制说明</summary>
    <div style="margin-top:6px;">
      系统遵循分布式身份管理机制，即：<br>
      1，用户掌控身份私钥，授权本地部署的节点使用身份。<br>
      2，本地部署的AI节点管理多用户相互隔离的数字空间。<br>
      3，上游社区节点保存加密身份副本用于追溯和自证。<br>
      在多方协作下共同保障隐私安全、身份可信及跨节点互认。以此构建“和而不同”的开源社区生态。<br>
    </div>
  </details>
</div>
'''

def update_after_identity_all(state):
    results = update_after_identity(state)
    results += get_all_user_default(state)
    return results

def update_after_identity(state):

    results = refresh_nav_bars(state)
    results += update_after_identity_sub(state)

    return results

def update_after_identity_sub(state):
    #[gallery_index, index_radio, gallery_index_stat, layer_method, layer_input_image, preset_store, preset_store_list, history_link, identity_introduce, configure_panel, admin_panel, p2p_panel, admin_link, system_params] + ip_types
    max_per_page = state["__max_per_page"]
    max_catalog = state["__max_catalog"]
    nickname = state["user"].get_nickname()
    user_did = state["user"].get_did()
    engine_type = state["engine_type"]
    logger.info(f'Session identity/当前身份: {nickname}({user_did}{", admin" if shared.token.is_admin(user_did) else ""}), session({state["__session"]})')
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog, user_did, engine_type)
    state.update({"__output_list": output_list})
    state.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})

    is_guest = shared.token.is_guest(user_did)
    is_admin = shared.token.is_admin(user_did)
    has_admin = check_admin_exists()
    is_privileged_guest = is_guest and (not has_admin)

    if is_admin or is_privileged_guest:
        comfyui_outputs = os.path.abspath(os.path.join(shared.token.get_path_in_user_dir(user_did, "outputs"), 'ComfyUI'))
        if not os.path.exists(comfyui_outputs):
            os.makedirs(comfyui_outputs)
        print(f"Setting ComfyUI outputs to: {comfyui_outputs}")
        comfyd.modify_variable({"outputs": comfyui_outputs})
        try:
            for arg in comfyd.comfyd_args:
                if len(arg) >= 2 and arg[0] == "--output-directory":
                    arg[1] = comfyui_outputs
                    print(f"Updated Comfyd startup args output directory to: {comfyui_outputs}")
                    break
        except Exception as e:
            print(f"Error updating comfyd startup args: {e}")

    results = [gr.update(choices=output_list, value=None), gr.update(visible=len(output_list)>0, open=False)]
    results += [state['__finished_nums_pages']]
    results += [gr.update(interactive=True if state["engine"]=='Fooocus' else False)] *2
    results += [gr.update(visible=False if 'preset_store' not in state else state['preset_store'])]
    results += [gr.Dataset.update(samples=get_preset_samples(user_did))]
    results += [update_history_link(user_did, state["local_access"])]
    results += [gr.update(visible=is_guest)]
    results += [gr.update(visible=True)]
    results += [gr.update(visible=is_admin or is_privileged_guest)]
    results += [gr.update(visible=is_admin or is_privileged_guest)]
    results += [gr.update(visible=False)]
    results += [gr.update(visible=is_admin or is_privileged_guest, value=update_comfyd_url(state))]
    results += update_topbar_js_params(state)
    ip_list = modules.flags.ip_list if state["engine"] in ['Fooocus', 'Flux', 'Kolors', 'Comfy', 'Wan', 'Qwen', 'Z-image']  else modules.flags.ip_list[:-1]
    ip_list = modules.flags.ip_list if state["engine"] in ['Fooocus', 'Flux', 'Kolors', 'Comfy', 'Wan', 'Qwen', 'Z-image']  else modules.flags.ip_list[:-1]
    ip_list = (ip_list[1:3] + ip_list[-1:]) if state["engine"] in ['Wan', 'Qwen', 'Z-image'] or state["task_method"] == 'flux2_aio_cn' else ip_list
    ip_list = (ip_list[:3] + ip_list[-1:]) if state["engine"]=='Comfy' and state["task_method"] == 'il_v_pre_aio' else ip_list
    default_controlnet_image_count = config.default_controlnet_image_count if state["engine"]=='Fooocus' else 4
    for image_count in range(default_controlnet_image_count):
        image_count += 1
        results.append(gr.update(choices=ip_list, value=config.default_ip_types[image_count]))

    return results

def update_size_and_hires_fix(image, uov_method, params_backend, hires_fix_stop, hires_fix_weight, hires_fix_blurred):
    size_image = update_upscale_size_of_image(image, uov_method)
    params_backend.update({'i2i_uov_hires_fix_s': hires_fix_stop})
    params_backend.update({'i2i_uov_hires_fix_w': hires_fix_weight})
    params_backend.update({'i2i_uov_hires_fix_blurred': hires_fix_blurred})
    vary_strength = -1
    vary_visible = False
    upscale_strength = -1
    upscale_visible = False
    if 'Upscale' in uov_method:
        upscale_visible = True
        upscale_strength = 0.2
    if 'Vary' in uov_method:
        vary_visible = True
        if 'Subtle' in uov_method:
            vary_strength = 0.5
        if 'Strong' in uov_method:
            vary_strength = 0.85
    if 'Hires.fix' in uov_method:
        vary_strength = 0.85
        vary_visible = True
    return gr.update(value=size_image), gr.update(visible='Hires.fix' in uov_method), gr.update(visible=vary_visible, value=vary_strength), gr.update(interactive=not 'Fast' in uov_method, visible=upscale_visible, value=upscale_strength)

def update_upscale_size_of_image(image, uov_method):
    if image is not None:
        H, W, C = util.HWC3(image).shape
    else:
        return ''
    match = re.search(r'\((?:Fast )?([\d.]+)x\)', uov_method)
    match_multiple = 1.0 if not match else float(match.group(1))
    match_multiple = match_multiple if match_multiple<4.0 else 4.0 
    width = int(W * match_multiple)
    height = int(H * match_multiple)

    return f'{W} x {H} | {width} x {height}'

def get_all_user_default(state):
    #[backfill_prompt, image_tools_checkbox, disable_preview, disable_intermediate_results, disable_seed_increment, save_final_enhanced_image_only, style_preview_checkbox]
    results = [ads.get_user_default("backfill_prompt", state, config.default_backfill_prompt)]
    results += [ads.get_user_default("image_tools_checkbox", state, True)]
    results += [ads.get_user_default("disable_preview", state, False)]
    results += [ads.get_user_default("disable_intermediate_results", state, False)]
    results += [ads.get_user_default("disable_seed_increment", state, False)]
    results += [ads.get_user_default("save_final_enhanced_image_only", state, False)]
    results += [ads.get_user_default("style_preview_checkbox", state)]
    results += [ads.get_user_default("generate_image_grid", state, False)]
    results += [ads.get_user_default("black_out_nsfw", state, config.default_black_out_nsfw)]
    results += [ads.get_user_default("save_metadata_to_images", state, config.default_save_metadata_to_images)]
    results += [ads.get_user_default("metadata_scheme", state, config.default_metadata_scheme)]
    results += [ads.get_user_default("no_model_modal_checkbox", state, False)]
    return results

def get_preferred_output_format(state_params):
    allowed = modules.flags.OutputFormat.list()
    user_did = None
    try:
        user = state_params.get("user", None) if isinstance(state_params, dict) else None
        user_did = user.get_did() if user is not None and hasattr(user, "get_did") else None
    except Exception:
        user_did = None
    if not user_did:
        try:
            user_did = shared.token.get_guest_did()
        except Exception:
            user_did = None

    preset_name = None
    try:
        preset_name = state_params.get("__preset", None) if isinstance(state_params, dict) else None
    except Exception:
        preset_name = None
    if not preset_name:
        preset_name = config.preset

    user_fmt = None
    try:
        user_session = state_params.get("__session", None) if isinstance(state_params, dict) else None
        ua_hash = state_params.get("ua_hash", None) if isinstance(state_params, dict) else None
        if user_session and ua_hash:
            raw_user_fmt = shared.token.get_local_vars("output_format", "Unknown", user_session, ua_hash)
            if isinstance(raw_user_fmt, str):
                raw_user_fmt = raw_user_fmt.strip()
            if raw_user_fmt and raw_user_fmt not in ["Unknown", "None"]:
                user_fmt = ads.convert_value(str(raw_user_fmt))
    except Exception:
        user_fmt = None

    preset_fmt = None
    try:
        if preset_name:
            preset_content = config.try_get_preset_content(str(preset_name), user_did)
            if isinstance(preset_content, dict):
                preset_fmt = preset_content.get("default_output_format", None)
    except Exception:
        preset_fmt = None

    fmt = preset_fmt if preset_fmt in allowed else user_fmt
    if fmt not in allowed:
        fmt = "jpeg"
    try:
        logger.debug(f"[OutputFormat] preset={preset_name}, preset_default={preset_fmt}, user={user_fmt}, final={fmt}")
    except Exception:
        pass
    return fmt

def apply_preferred_output_format(state_params):
    return gr.update(value=get_preferred_output_format(state_params))

def restore_all_defaults(state_params):
    user_keys = ["__lang", "__theme", "backfill_prompt", "image_tools_checkbox", "disable_preview", "disable_intermediate_results", "disable_seed_increment", "save_final_enhanced_image_only", "style_preview_checkbox", "generate_image_grid", "black_out_nsfw", "save_metadata_to_images", "metadata_scheme", "no_model_modal_checkbox", "output_format"]
    admin_keys = ["comfyd_active_checkbox", "fast_comfyd_checkbox", "cache_clear_on_finish_checkbox", "reserved_vram", "cache_ram_enable", "cache_ram", "minicpm_checkbox", "minicpm_version", "advanced_logs", "wavespeed_strength", "translation_methods", "p2p_active_checkbox", "p2p_remote_process", "p2p_in_did_list", "p2p_out_did_list", "no_welcome_checkbox", "missing_model_filter_checkbox"]

    try:
        user = state_params.get("user", None) if isinstance(state_params, dict) else None
        user_did = user.get_did() if user is not None and hasattr(user, "get_did") else None
    except Exception:
        user_did = None
    try:
        logger.info(f"[RestoreDefaults] confirm: session={state_params.get('__session', None)}, ua_hash={state_params.get('ua_hash', None)}, did={user_did}")
    except Exception:
        pass

    user_session = state_params.get("__session", None) if isinstance(state_params, dict) else None
    ua_hash = state_params.get("ua_hash", None) if isinstance(state_params, dict) else None
    if not user_session or not ua_hash:
        try:
            gr.Info("Restore failed: missing session/ua_hash (refresh page and retry).")
        except Exception:
            pass
        outputs_len = 1 + 4 + len(get_all_user_default(state_params)) + len(admin_keys) + 1
        return [gr.update(visible=True)] + [gr.update()] * (outputs_len - 1)

    for key in user_keys:
        try:
            shared.token.set_local_vars(key, "Default", user_session, ua_hash)
        except Exception:
            pass
        try:
            ads.cache_vars.pop(f"{user_session}_{key}", None)
        except Exception:
            pass

    for key in admin_keys:
        try:
            shared.token.set_local_admin_vars(key, "", user_session, ua_hash)
        except Exception:
            pass
        try:
            ads.cache_vars.pop(f"admin_{key}", None)
        except Exception:
            pass

    try:
        if "__lang" in state_params:
            del state_params["__lang"]
        if "__theme" in state_params:
            del state_params["__theme"]
    except Exception:
        pass

    try:
        gr.Info("Defaults restored (including admin settings).")
    except Exception:
        pass

    lang_internal = ads.get_user_default("__lang", state_params, args_manager.args.language)
    theme_internal = ads.get_user_default("__theme", state_params, args_manager.args.theme)
    user_defaults = [gr.update(value=v) for v in get_all_user_default(state_params)]

    admin_values_new = []
    for key in admin_keys:
        v = ads.get_admin_default(key)
        if key == "comfyd_active_checkbox":
            if args_manager.args.disable_comfyd or args_manager.args.disable_backend:
                v = False
        elif key == "translation_methods":
            try:
                if v not in modules.flags.translation_methods:
                    v = config.default_translation_methods
            except Exception:
                pass
        elif key == "p2p_remote_process":
            try:
                if v not in ["Disable", "out", "in"]:
                    v = "Disable"
            except Exception:
                pass
        admin_values_new.append(v)

    admin_updates = []
    for k, v in zip(admin_keys, admin_values_new):
        if k == "cache_ram":
            try:
                cache_enable = bool(admin_values_new[admin_keys.index("cache_ram_enable")])
            except Exception:
                cache_enable = True
            admin_updates.append(gr.update(value=v, interactive=cache_enable))
        elif k == "p2p_remote_process":
            try:
                p2p_enable = bool(admin_values_new[admin_keys.index("p2p_active_checkbox")])
            except Exception:
                p2p_enable = True
            admin_updates.append(gr.update(value=v, interactive=p2p_enable))
        else:
            admin_updates.append(gr.update(value=v))

    fmt = get_preferred_output_format(state_params)
    return [gr.update(visible=False), gr.update(), gr.update(value=modules.flags.language_radio(lang_internal)), gr.update(value=theme_internal), gr.update()] + user_defaults + admin_updates + [gr.update(value=fmt)]

def get_all_admin_default(currunt_value):
    admin_keys = ['comfyd_active_checkbox', 'fast_comfyd_checkbox', 'cache_clear_on_finish_checkbox', 'reserved_vram', 'cache_ram_enable', 'cache_ram', 'minicpm_checkbox', 'minicpm_version', 'advanced_logs', 'wavespeed_strength', 'translation_methods', 'p2p_active_checkbox', "p2p_remote_process", "p2p_in_did_list", "p2p_out_did_list", "no_welcome_checkbox", "missing_model_filter_checkbox"]
    result = []
    for i, admin_key in enumerate(admin_keys):
        admin_value = ads.get_admin_default(admin_key)

        if admin_value == 'None':
            result.append(gr.update(interactive=False))
            continue
        if admin_value == currunt_value[i]:
            result.append(gr.update())
        else:
            if admin_key in ["p2p_in_did_list", "p2p_out_did_list"]:
                result.append(gr.update(value=admin_value))
            else:
                if admin_key == 'comfyd_active_checkbox':
                    admin_value = 'False' if args_manager.args.disable_comfyd or args_manager.args.disable_backend else admin_value
                elif admin_key == 'translation_methods' and admin_value not in modules.flags.translation_methods:
                    admin_value = config.default_translation_methods
                result.append(gr.update(interactive=True, value=admin_value))
    return result


from transformers import CLIPTokenizer
import shutil

CLIP_TOKENIZER_URLS = [
    "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip_vision/clip-vit-large-patch14/merges.txt",
    "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip_vision/clip-vit-large-patch14/special_tokens_map.json",
    "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip_vision/clip-vit-large-patch14/tokenizer_config.json",
    "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip_vision/clip-vit-large-patch14/vocab.json"
]

cur_clip_path = os.path.join(config.paths_clip_vision[0], "clip-vit-large-patch14")

def check_clip_validity(path):
    required_files = ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
    if not os.path.exists(path):
        return False
    for f in required_files:
        if not os.path.exists(os.path.join(path, f)) or os.path.getsize(os.path.join(path, f)) == 0:
            return False
    return True

if not check_clip_validity(cur_clip_path):
    print(f"CLIP directory {cur_clip_path} is missing or invalid.")
    if os.path.exists(cur_clip_path):
        try:
            shutil.rmtree(cur_clip_path)
            print(f"Removed invalid directory: {cur_clip_path}")
        except Exception as e:
            print(f"Failed to remove invalid directory: {e}")

    org_clip_path = os.path.join(shared.root, 'models/clip_vision/clip-vit-large-patch14')
    if check_clip_validity(org_clip_path):
        print(f"Copying CLIP from backup: {org_clip_path}")
        try:
            shutil.copytree(org_clip_path, cur_clip_path)
        except Exception as e:
             print(f"Failed to copy from backup: {e}")
    else:
        print(f"Downloading CLIP tokenizer files to {cur_clip_path}...")
        try:
            os.makedirs(cur_clip_path, exist_ok=True)
            for url in CLIP_TOKENIZER_URLS:
                file_name = url.split('/')[-1]
                print(f"Downloading {file_name}...")
                load_file_from_url(url=url, model_dir=cur_clip_path, file_name=file_name)
        except Exception as e:
            print(f"Failed to download CLIP: {e}")

try:
    tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
except Exception as e:
    print(f"Error loading tokenizer from {cur_clip_path}: {e}")
    tokenizer = None
 
def remove_tokenizer():
    global tokenizer

    if 'tokenizer' in globals():
        del tokenizer
    return

def prompt_token_prediction(text, style_selections):
    global tokenizer, cur_clip_path
    if 'tokenizer' not in globals():
        globals()['tokenizer'] = None
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
    return len(tokenizer.tokenize(text))

#system_message = get_system_message()

def stop_comfyd_background(comfyd_active_checkbox):
    if comfyd_active_checkbox:
        threading.Thread(target=comfyd.stop).start()
