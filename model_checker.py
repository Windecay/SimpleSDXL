import os
import re
import sys
import time
import requests
import queue
import platform
import hashlib
from tqdm import tqdm
from colorama import init, Fore, Style
import threading
import atexit
import json
from collections import defaultdict
from multiprocessing import current_process
DEFAULT_DOWNLOAD_PREFIX = "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/"
HF_DOWNLOAD_PREFIX = "https://huggingface.co/metercai/SimpleSDXL2/resolve/main/"
CURRENT_DOWNLOAD_PREFIX = os.getenv('CURRENT_DOWNLOAD_PREFIX', DEFAULT_DOWNLOAD_PREFIX)
current_source = "ModelScope魔搭国内源" if CURRENT_DOWNLOAD_PREFIX == DEFAULT_DOWNLOAD_PREFIX else "HuggingFace拥抱脸国外源"

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if current_process().name != "MainProcess":
    root_dir = os.path.dirname(os.path.dirname(root_dir))
elif platform.system() == 'Windows':
    root_dir = os.path.dirname(root_dir)

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print_colored(f"√创建目录: {directory}", Fore.GREEN)
        except Exception as e:
            print_colored(f"×创建目录失败: {directory}, 错误: {e}", Fore.RED)
            return False
    return True

def load_model_paths():
    global simplemodels_root

    config_path = os.path.normpath(os.path.join(root_dir, "users", "config.txt"))
    path_mapping = {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        models_root = config.get("path_models_root", None)
        if models_root:
            simplemodels_root = os.path.abspath(os.path.join(script_dir, models_root)) if not os.path.isabs(models_root) else models_root
        else:
            simplemodels_root = os.path.normpath(os.path.join(root_dir, "SimpleModels"))
        path_mapping = {
            "checkpoints": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_checkpoints", [])],
            "loras": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                    for p in config.get("path_loras", [])],
            "controlnet": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_controlnet", [])],
            "embeddings": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_embeddings")] if isinstance(config.get("path_embeddings"), str)
                                else config.get("path_embeddings", []))],
            "vae_approx": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_vae_approx")] if isinstance(config.get("path_vae_approx"), str)
                                else config.get("path_vae_approx", []))],
            "vae": [os.path.abspath(os.path.join(script_dir, p))
                    for p in ([config.get("path_vae")] if isinstance(config.get("path_vae"), str)
                            else config.get("path_vae", []))],
            "upscale_models": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_upscale_models")] if isinstance(config.get("path_upscale_models"), str)
                            else config.get("path_upscale_models", []))],
            "inpaint": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_inpaint", [])],
            "clip": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_clip")] if isinstance(config.get("path_clip"), str)
                            else config.get("path_clip", []))],
            "clip_vision": [os.path.abspath(os.path.join(script_dir, p))
                        for p in ([config.get("path_clip_vision")] if isinstance(config.get("path_clip_vision"), str)
                            else config.get("path_clip_vision", []))],
            "fooocus_expansion": [os.path.abspath(os.path.join(script_dir, config.get("path_fooocus_expansion", "")))],
            "llms": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_llms", [os.path.join(simplemodels_root, "llms")])
                                if isinstance(config.get("path_llms"), list)
                                else [config.get("path_llms") or os.path.join(simplemodels_root, "llms")])],
            "LLM": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_llm", config.get("path_LLM", [os.path.join(simplemodels_root, "LLM")]))
                                if isinstance(config.get("path_llm", config.get("path_LLM")), list)
                                else [config.get("path_llm", config.get("path_LLM")) or os.path.join(simplemodels_root, "LLM")])],
            "safety_checker": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_safety_checker", [])
                            if isinstance(config.get("path_safety_checker"), list)
                            else [config.get("path_safety_checker", "")])],
            "unet": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_unet", [])
                            if isinstance(config.get("path_unet"), list)
                            else [config.get("path_unet", "")])],
            "rembg": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_rembg", [])
                            if isinstance(config.get("path_rembg"), list)
                            else [config.get("path_rembg", "")])],
            "layer_model": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_layer_model", [])
                                if isinstance(config.get("path_layer_model"), list)
                                else [config.get("path_layer_model", "")])],
            "diffusers": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in config.get("path_diffusers", [])],
            "ipadapter": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_ipadapter", [])
                            if isinstance(config.get("path_ipadapter"), list)
                            else [config.get("path_ipadapter", "")])],
            "pulid": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_pulid", [])
                        if isinstance(config.get("path_pulid"), list)
                        else [config.get("path_pulid", "")])],
            "insightface": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_insightface", [])
                                if isinstance(config.get("path_insightface"), list)
                                else [config.get("path_insightface", "")])],
            "style_models": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_style_models", [])
                                if isinstance(config.get("path_style_models"), list)
                                else [config.get("path_style_models", "")])],
            "configs": [os.path.abspath(os.path.join(simplemodels_root, "configs"))],
            "prompt_expansion": [os.path.abspath(os.path.join(simplemodels_root, "prompt_expansion"))],
            "model_patches": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_model_patches", [os.path.join(simplemodels_root, "model_patches")])
                                if isinstance(config.get("path_model_patches"), list)
                                else [config.get("path_model_patches") or os.path.join(simplemodels_root, "model_patches")])],
            "audio_encoders": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_audio_encoders", [os.path.join(simplemodels_root, "audio_encoders")])
                                    if isinstance(config.get("path_audio_encoders"), list)
                                    else [config.get("path_audio_encoders") or os.path.join(simplemodels_root, "audio_encoders")])],
            "text_encoders": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_text_encoders", [os.path.join(simplemodels_root, "text_encoders")])
                                if isinstance(config.get("path_text_encoders"), list)
                                else [config.get("path_text_encoders") or os.path.join(simplemodels_root, "text_encoders")])],
            "lsnet": [os.path.join(simplemodels_root, "lsnet")],
            "kaloscope": [os.path.join(simplemodels_root, "lsnet", "kaloscope")],
            "detection": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_detection", [os.path.join(simplemodels_root, "detection")])
                                if isinstance(config.get("path_detection"), list)
                                else [config.get("path_detection") or os.path.join(simplemodels_root, "detection")])],
            "diffusion_models": [os.path.abspath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p
                        for p in (config.get("path_diffusion_models", [os.path.join(simplemodels_root, "diffusion_models")])
                                if isinstance(config.get("path_diffusion_models"), list)
                                else [config.get("path_diffusion_models") or os.path.join(simplemodels_root, "diffusion_models")])],
            "ultralytics": [os.path.join(simplemodels_root, "ultralytics")],
            "bbox": [os.path.join(simplemodels_root, "ultralytics", "bbox")],
            "segm": [os.path.join(simplemodels_root, "ultralytics", "segm")],
            "SDPose_OOD": [os.path.join(simplemodels_root, "SDPose_OOD")],
            "yolo": [os.path.join(simplemodels_root, "yolo")],
            "jina_clip": [os.path.join(simplemodels_root, "jina_clip")],
            "gemma3": [os.path.join(simplemodels_root, "gemma3")],
            "nlf": [os.path.join(simplemodels_root, "nlf")],
            "SEEDVR2": [os.path.join(simplemodels_root, "SEEDVR2")],
        }

    except Exception as e:
        print(f"{Fore.YELLOW}△配置文件加载失败: {e}，使用默认路径{Style.RESET_ALL}")
        # 设置默认的simplemodels_root
        simplemodels_root = os.path.normpath(os.path.join(root_dir, "SimpleModels"))
        path_mapping = {
            "checkpoints": [os.path.join(simplemodels_root, "checkpoints")],
            "loras": [os.path.join(simplemodels_root, "loras")],
            "controlnet": [os.path.join(simplemodels_root, "controlnet")],
            "embeddings": [os.path.join(simplemodels_root, "embeddings")],
            "vae_approx": [os.path.join(simplemodels_root, "vae_approx")],
            "vae": [os.path.join(simplemodels_root, "vae")],
            "upscale_models": [os.path.join(simplemodels_root, "upscale_models")],
            "inpaint": [os.path.join(simplemodels_root, "inpaint")],
            "clip": [os.path.join(simplemodels_root, "clip")],
            "clip_vision": [os.path.join(simplemodels_root, "clip_vision")],
            "fooocus_expansion": [os.path.join(simplemodels_root, "prompt_expansion", "fooocus_expansion")],
            "llms": [os.path.join(simplemodels_root, "llms")],
            "LLM": [os.path.join(simplemodels_root, "LLM")],
            "safety_checker": [os.path.join(simplemodels_root, "safety_checker")],
            "unet": [os.path.join(simplemodels_root, "unet")],
            "rembg": [os.path.join(simplemodels_root, "rembg")],
            "layer_model": [os.path.join(simplemodels_root, "layer_model")],
            "diffusers": [os.path.join(simplemodels_root, "diffusers")],
            "ipadapter": [os.path.join(simplemodels_root, "ipadapter")],
            "pulid": [os.path.join(simplemodels_root, "pulid")],
            "insightface": [os.path.join(simplemodels_root, "insightface")],
            "style_models": [os.path.join(simplemodels_root, "style_models")],
            "configs": [os.path.normpath(os.path.join(simplemodels_root, "configs"))],
            "prompt_expansion": [os.path.normpath(os.path.join(simplemodels_root, "prompt_expansion"))],
            "model_patches": [os.path.join(simplemodels_root, "model_patches")],
            "audio_encoders": [os.path.join(simplemodels_root, "audio_encoders")],
            "text_encoders": [os.path.join(simplemodels_root, "text_encoders")],
            "lsnet": [os.path.join(simplemodels_root, "lsnet")],
            "kaloscope": [os.path.join(simplemodels_root, "lsnet", "kaloscope")],
            "detection": [os.path.join(simplemodels_root, "detection")],
            "diffusion_models": [os.path.join(simplemodels_root, "diffusion_models")],
            "ultralytics": [os.path.join(simplemodels_root, "ultralytics")],
            "bbox": [os.path.join(simplemodels_root, "ultralytics", "bbox")],
            "segm": [os.path.join(simplemodels_root, "ultralytics", "segm")],
            "SDPose_OOD": [os.path.join(simplemodels_root, "SDPose_OOD")],
            "yolo": [os.path.join(simplemodels_root, "yolo")],
            "jina_clip": [os.path.join(simplemodels_root, "jina_clip")],
            "gemma3": [os.path.join(simplemodels_root, "gemma3")],
            "nlf": [os.path.join(simplemodels_root, "nlf")],
            "SEEDVR2": [os.path.join(simplemodels_root, "SEEDVR2")],
        }

    for key in path_mapping:
        path_mapping[key] = [
            os.path.abspath(p) if not os.path.isabs(p) else p
            for p in path_mapping[key]
        ]
        path_mapping[key] = list(set(path_mapping[key]))

    for key, paths in path_mapping.items():
        for path in paths:
            ensure_directory_exists(path)

    return path_mapping

def cleanup():
    if os.path.exists("downloadlist.txt"):
        os.remove("downloadlist.txt")
        print("已删除 'downloadlist.txt' 文件。")
    if os.path.exists("缺失模型下载链接.txt"):
        os.remove("缺失模型下载链接.txt")
        print("已删除 '缺失模型下载链接.txt' 文件。")
atexit.register(cleanup)

init(autoreset=True, strip=False, convert=False)

class DownloadStatus:
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=filename,
            position=0,
            leave=False,         # 完成后保留进度条（显示100%）
            dynamic_ncols=True,  # 动态列宽
            file=sys.stdout,     # 输出到 stdout
            miniters=1,          # 每次迭代都更新
            mininterval=0.05,    # 缩短更新间隔到 0.05 秒
            disable=False,       # 明确不禁用
            ncols=100            # 固定列宽
        )

def print_colored(text, color=Fore.WHITE):
    print(f"{color}{text}{Style.RESET_ALL}")

def check_python_embedded():
    python_exe = sys.executable
    print(f"Python解析器路径: {python_exe}")

    if platform.system() =='Windows' and "python_embeded" not in python_exe.lower():
        print_colored("×当前 Python 解释器不在 python_embeded 目录中，请检查运行环境", Fore.RED)
        print("按任意键继续。", flush=True)
        input()
        sys.exit(1)

def check_script_file():
    script_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SimpleSDXL", "entry_with_update.py")

    if os.path.exists(script_file):
        print_colored("√找到主程序目录", Fore.GREEN)
    else:
        print_colored("×未找到主程序目录，请检查脚本位置", Fore.RED)
        print("按任意键继续。", flush=True)
        input()
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(script_file))
    directory_level = len(base_dir.split(os.sep))

    if directory_level <= 2:
        print_colored("×主程序目录层级不足，可能会导致脚本结果有误。请按照安装视频指引先建立SimpleAI主文件夹", Fore.RED)
    else: 
        print_colored("√主程序目录层级验证通过", Fore.GREEN)

    paths_to_check = [
        ("当前脚本路径", os.path.abspath(__file__)),
        ("主程序目录", base_dir),
        ("入口文件路径", script_file)
    ]

    has_space = False
    for desc, path in paths_to_check:
        if ' ' in path:
            print_colored(f"!警告：{desc}包含空格 -> {path}", Fore.YELLOW)
            has_space = True

    if has_space:
        print_colored("!路径包含空格可能导致程序异常，建议将SimpleAI安装到无空格路径（如D:\\SimpleAI）", Fore.YELLOW)
        time.sleep(10)

def get_total_virtual_memory():
    try:
        import psutil
        virtual_mem = psutil.virtual_memory().total
        swap_mem = psutil.swap_memory().total
        total_virtual_memory = virtual_mem + swap_mem
        return total_virtual_memory
    except ImportError:
        print_colored("无法导入 psutil 模块，跳过内存检查", Fore.YELLOW)
        return None
    except Exception as e:
        print_colored(f"无法获取系统虚拟内存，可能是性能计数器未开启或其他问题。\n错误详情: {e}", Fore.YELLOW)
        print_colored("请参考https://learn.microsoft.com/zh-cn/troubleshoot/windows-server/performance/rebuild-performance-counter-library-values重新启用系统性能计数器，或忽略此警告继续。", Fore.YELLOW)
        return None

def check_virtual_memory(total_virtual):
    if total_virtual is None:
        print_colored("跳过虚拟内存检查。", Fore.YELLOW)
        return
    total_gb = total_virtual / (1024 ** 3)
    if total_gb < 40:
        print_colored("警告：系统虚拟内存小于40GB，会禁用部分预置包，请参考安装视频教程设置系统虚拟内存。", Fore.YELLOW)
    else:
        print_colored("√系统虚拟内存充足", Fore.GREEN)
    print(f"系统总虚拟内存: {total_gb:.2f} GB")

def find_simplemodels_dir(start_path):
    current_dir = start_path
    while current_dir != os.path.dirname(current_dir):
        simplemodels_path = os.path.join(current_dir, "SimpleModels")
        if os.path.isdir(simplemodels_path):
            return simplemodels_path
        current_dir = os.path.dirname(current_dir)
    return None

def find_users_dir(start_path):
    current_dir = start_path
    while current_dir != os.path.dirname(current_dir):
        users_path = os.path.join(current_dir, "users")
        if os.path.isdir(users_path):
            return users_path
        current_dir = os.path.dirname(current_dir)
    return None

def normalize_path(path):
    path_mapping = load_model_paths()

    path_parts = path.split('/')
    if len(path_parts) < 2:
        return os.path.abspath(path)

    path_type = path_parts[0]
    filename = '/'.join(path_parts[1:])

    sorted_dirs = sorted(
        path_mapping.get(path_type, []),
        key=lambda x: (
            0 if "SimpleModels" in x else
            1 if any(part == "models" for part in x.split(os.sep)) else
            2,
            x
        )
    )

    for base_dir in sorted_dirs:
        full_path = os.path.join(base_dir, filename)
        return os.path.abspath(full_path)

def typewriter_effect(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
        
    print()
def print_instructions():
    print()
    print(f"{Fore.GREEN}★★★★★{Style.RESET_ALL}安装视频教程{Fore.YELLOW}https://www.bilibili.com/video/BV1ddkdYcEWg/{Style.RESET_ALL}{Fore.GREEN}★★★★★{Style.RESET_ALL}{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print()
    print(f"{Fore.GREEN}★{Style.RESET_ALL}攻略地址飞书文档:{Fore.YELLOW}https://acnmokx5gwds.feishu.cn/wiki/QK3LwOp2oiRRaTkFRhYcO4LonGe{Style.RESET_ALL}文章无权限即为未编辑完毕。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}稳速生图指南:Nvidia显卡驱动选择最新版驱动,驱动类型最好为Studio。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}在遇到生图速度断崖式下降或者爆显存OutOfMemory时,提高{Fore.GREEN}预留显存功能{Style.RESET_ALL}的数值至（1~2）{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}打开默认浏览器设置，关闭GPU加速、或图形加速的选项。{Fore.GREEN}★{Style.RESET_ALL}大内存(64+)与固态硬盘存放模型有助于减少模型加载时间。{Fore.GREEN}★{Style.RESET_ALL}")
    time.sleep(0.1)
    print(f"{Fore.GREEN}★{Style.RESET_ALL}疑难杂症进QQ群求助：1005085136{Fore.GREEN}★{Style.RESET_ALL}脚本：✿   冰華 |版本:26.01.17{Fore.GREEN}★{Style.RESET_ALL}")
    print()
    time.sleep(0.1)
    
def get_unique_filename(file_path, extension=".corrupted"):
    base = file_path + extension
    counter = 1
    while os.path.exists(base):
        base = f"{file_path}{extension}_{counter}"
        counter += 1
    return base
def get_actual_file_path(file_path):
    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    url_match = re.search(url_pattern, file_path)

    if url_match:
        local_path = file_path.split(url_match.group(0))[0].rstrip('/')
        url_file_name = os.path.basename(url_match.group(0))
        actual_file_path = os.path.join(local_path, url_file_name)
        return os.path.normpath(actual_file_path)
    else:
        return os.path.normpath(file_path)
def validate_files(packages):
    cleanup()
    path_mapping = load_model_paths()
    print_colored(f">>>>>>默认模型根目录为：{simplemodels_root}<<<<<<", Fore.YELLOW)
    print()
    # 根据GPU架构过滤packages
    filtered_packages = filter_packages_by_gpu_arch(packages)

    # 获取GPU架构信息并显示
    gpu_arch = get_gpu_arch_str()
    print_colored(f"当前GPU架构: {gpu_arch}, 已根据架构过滤预置包", Fore.CYAN)

    download_files = {}
    missing_package_names = []
    package_percentages = {}
    package_sizes = {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for package_key, package_info in filtered_packages.items():
        package_name = package_info["name"]
        package_note = package_info.get("note", "")
        files_and_sizes = package_info["files"]
        download_links = package_info["download_links"]


        total_size = sum([size for _, size in files_and_sizes])
        total_size_gb = total_size / (1024 ** 3)
        non_missing_size = 0

        print(f"－－－－－－－", end='')
        time.sleep(0.1)
        print(f"校验{package_name}文件－－－－{package_note}")

        missing_files = []
        size_mismatch_files = []
        case_mismatch_files = []

        for expected_path, expected_size in files_and_sizes:
            expected_filename = os.path.basename(expected_path) 
            path_parts = expected_path.split('/')
            path_type = path_parts[0] if len(path_parts) > 0 else ''
            sub_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''

            search_dirs = sorted(
                path_mapping.get(path_type, []),
                key=lambda x: (
                    0 if "SimpleModels" in x else
                    1 if any(part == "models" for part in x.split(os.sep)) else
                    2,
                    x
                )
            )
            if not search_dirs:
                simplemodels_default = os.path.join(root, "SimpleModels")
                search_dirs = [os.path.join(simplemodels_default, path_type)]

            found = False
            actual_dir = None
            url_pattern = r'https?://[^\s/$.?#].[^\s]*'
            url_match = re.search(url_pattern, expected_path)
            if url_match:
                local_dir = expected_path.split(url_match.group(0))[0].rstrip('/')
                file_name = os.path.basename(url_match.group(0))

                for base_dir in search_dirs:
                    actual_full_path = os.path.join(base_dir, local_dir.replace(path_type, "", 1).lstrip('/'), file_name)

                    actual_full_path = os.path.normpath(actual_full_path)

                    if os.path.exists(actual_full_path):
                        actual_dir = os.path.dirname(actual_full_path)
                        found = True
                        expected_path = os.path.join(local_dir, file_name)
                        break
                # if found:
                #     continue
            for base_dir in search_dirs:
                full_path = os.path.join(base_dir, sub_path) if sub_path else os.path.join(base_dir, os.path.basename(expected_path))
                if os.path.exists(full_path):
                    actual_dir = os.path.dirname(full_path)
                    found = True
                    break

            if not found:
                missing_files.append((expected_path, expected_size))
                continue

            try:
                directory_listing = os.listdir(actual_dir)
            except Exception as e:
                print(f"{Fore.RED}目录访问错误: {actual_dir} - {str(e)}{Style.RESET_ALL}")
                missing_files.append((expected_path, expected_size))
                continue

            expected_filename = os.path.basename(expected_path)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)
            directory_listing = os.listdir(actual_dir)
            actual_filename = next((f for f in directory_listing if f.lower() == expected_filename.lower()), None)

            if actual_filename is None:
                missing_files.append((expected_path, expected_size))
            elif actual_filename != expected_filename:
                case_mismatch_files.append((os.path.join(actual_dir, actual_filename), expected_filename))
            else:
                actual_size = os.path.getsize(os.path.join(actual_dir, actual_filename))
                if actual_size != expected_size:
                    size_mismatch_files.append((os.path.join(actual_dir, actual_filename), actual_size, expected_size))
                else:
                    non_missing_size += expected_size
        obsolete_files = []
        MODEL_PATHS_TO_SCAN = [
        os.path.join(simplemodels_root, "checkpoints"),
        os.path.join(simplemodels_root, "loras"),
        os.path.join(simplemodels_root, "controlnet"),
        os.path.join(simplemodels_root, "ipadapter"),
        os.path.join(simplemodels_root, "diffusers"),
        os.path.join(simplemodels_root, "model_patches"),
        os.path.join(simplemodels_root, "inpaint"),
        os.path.join(simplemodels_root, "clip"),
        os.path.join(simplemodels_root, "clip_vision"),
        os.path.join(simplemodels_root, "fooocus_expansion"),
        os.path.join(simplemodels_root, "llms"),
        os.path.join(simplemodels_root, "LLM"),
        os.path.join(simplemodels_root, "safety_checker"),
        os.path.join(simplemodels_root, "unet"),
        os.path.join(simplemodels_root, "layer_model"),
        os.path.join(simplemodels_root, "pulid"),
        os.path.join(simplemodels_root, "text_encoders"),
        os.path.join(simplemodels_root, "lsnet"),
        os.path.join(simplemodels_root, "kaloscope"),
        os.path.join(simplemodels_root, "diffusion_models"),
        os.path.join(simplemodels_root, "SDPose_OOD"),
        os.path.join(simplemodels_root, "jina_clip"),
        os.path.join(simplemodels_root, "gemma3"),
        os.path.join(simplemodels_root, "nlf"),
        os.path.join(simplemodels_root, "SEEDVR2"),
        os.path.join(simplemodels_root, "LLM", "Qwen3-VL-4B-Instruct-abliterated"),
        ]
        for model_root in MODEL_PATHS_TO_SCAN:
            if not os.path.exists(model_root):
                continue
            for scan_root, _, files in os.walk(model_root):
                for file in files:
                    if file.lower() in [x.lower() for x in OBSOLETE_MODELS]:
                        full_path = os.path.join(scan_root, file)
                        obsolete_files.append(full_path)


        if total_size > 0:
            non_missing_percentage = (non_missing_size / total_size) * 100
            package_percentages[package_name] = non_missing_percentage
            package_sizes[package_name] = total_size_gb

        if case_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件名大小写不匹配，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_filename in case_mismatch_files:
                print(f"文件: {normalize_path(file)}")
                time.sleep(0.1)
                print(f"正确文件名: {expected_filename}")
                
                corrected_file_path = os.path.join(os.path.dirname(file), expected_filename)
                os.rename(file, corrected_file_path)
                print(f"{Fore.GREEN}文件名已更正为: {expected_filename}{Style.RESET_ALL}")

        if size_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。{Style.RESET_ALL}")
            for file, actual_size, expected_size in size_mismatch_files:
                normalized_path = normalize_path(file)
                print(f"{normalized_path} 当前大小={actual_size}, 预期大小={expected_size}")
                time.sleep(0.1)
                
                corrupted_file_path = get_unique_filename(file)
                os.rename(file, corrupted_file_path)
                print(f"{Fore.YELLOW}文件已重命名为: {normalize_path(corrupted_file_path)}（大小不匹配）{Style.RESET_ALL}")
                
                relative_path = os.path.relpath(file, root).replace(os.sep, '/')
                download_files[relative_path] = expected_size
                if package_name not in missing_package_names:
                    missing_package_names.append(package_name)

        if missing_files:
            print(f"{Fore.RED}×{package_name}有文件缺失，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_size in missing_files:
                url_pattern = r'https?://[^\s/$.?#].[^\s]*'
                url_match = re.search(url_pattern, file)
                if url_match:
                    print(normalize_path(file.split(url_match.group(0))[0] + os.path.basename(url_match.group(0))))
                else:
                    print(normalize_path(file))
                download_files[file] = expected_size
            if package_name not in missing_package_names:
                missing_package_names.append(package_name)

            if package_info["download_links"]:
                print(f"{Fore.YELLOW}下载链接(若为压缩包，则参考安装视频流程安装):{Style.RESET_ALL}")
                for link in package_info["download_links"]:
                    print(f"{Fore.YELLOW}{link}{Style.RESET_ALL}")
        if not missing_files and not size_mismatch_files and not case_mismatch_files:
            print(f"{Fore.GREEN}√{package_name}文件全部验证通过{Style.RESET_ALL}")


    if missing_package_names:
        print(f"{Fore.RED}△以下模型包缺失文件，请选择你想要的模块下载：{Style.RESET_ALL}")
        for package_name in missing_package_names:
            percentage = package_percentages.get(package_name, 0)
            total_size_gb = package_sizes.get(package_name, 0)

            missing_size_gb = total_size_gb * (1 - (percentage / 100))
            print(f"- {package_name} - 总大小：{total_size_gb:.2f}GB，完整度：{percentage:.2f}%，尚需下载：{missing_size_gb:.2f}GB")
    if obsolete_files:
        # 新增空间计算
        total_obsolete_size = 0
        for file in obsolete_files:
            try:
                total_obsolete_size += os.path.getsize(file)
            except:
                pass
        print(f"\n{Fore.YELLOW}△发现以下可删除的废弃模型：{Style.RESET_ALL}")
        for file in obsolete_files:
            print(f"  {file}")
        # 新增空间显示
        print(f"{Fore.CYAN}※这些模型已被新版替代，可节省空间: {total_obsolete_size/1024/1024/1024:.2f}GB (按0+回车清理时选择删除){Style.RESET_ALL}")

    sorted_download_files = sorted(download_files.items(), key=lambda x: x[1])

    if sorted_download_files:
        with open("downloadlist.txt", "w") as f1, open("缺失模型下载链接.txt", "w") as f2:
            for file, size in sorted_download_files:
                url_pattern = r'https?://[^\s/$.?#].[^\s]*'
                url_match = re.search(url_pattern, file)

                if url_match:
                    link = url_match.group(0)
                else:
                    link = f"{CURRENT_DOWNLOAD_PREFIX}SimpleModels/{file.split('SimpleModels/')[-1]}"

                f1.write(f"{link},{size}\n")
                f2.write(f"{link}\n")
        print(f"{Fore.YELLOW}>>>问题文件的文件下载链接已保存到 '缺失模型下载链接.txt'。<<<<<<<<<<<<<<<<<<<<<{Style.RESET_ALL}")

def delete_partial_files():
    global OBSOLETE_MODELS
    try:
        path_mapping = load_model_paths()
    except Exception as e:
        print(f"{Fore.RED}△路径配置加载失败: {str(e)}{Style.RESET_ALL}")
        return

    scan_categories = [
        'checkpoints', 'loras', 'controlnet', 'embeddings',
        'vae_approx', 'vae', 'upscale_models', 'inpaint', "ipadapter",
        'clip', 'clip_vision', 'llms', 'LLM', 'unet', 'diffusers', 'model_patches'
    ]

    scan_dirs = []
    for category in scan_categories:
        scan_dirs.extend(path_mapping.get(category, []))
    
    default_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "SimpleModels"
    ))
    if default_dir not in scan_dirs:
        scan_dirs.append(default_dir)

    total_size = 0
    files_found = False
    files_to_delete = []
    obsolete_files_found = []  # 新增废弃文件存储

    for model_dir in scan_dirs:
        if not os.path.exists(model_dir):
            continue

        print(f"{Fore.CYAN}△扫描目录: {model_dir}{Style.RESET_ALL}")

        for root, _, files in os.walk(model_dir):
            for file in files:
                if ".partial" in file or ".corrupted" in file:
                    file_path = os.path.join(root, file)
                    files_found = True
                    files_to_delete.append(file_path)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
                if file in OBSOLETE_MODELS:  # 精确文件名匹配
                    file_path = os.path.join(root, file)
                    obsolete_files_found.append(file_path)
                    files_found = True
    if files_found:
        print(f"{Fore.YELLOW}△以下未下载完或损坏的文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")

        obsolete_total = sum(os.path.getsize(f) for f in obsolete_files_found if os.path.exists(f))

        if obsolete_files_found:
            print(f"\n{Fore.YELLOW}△以下废弃模型文件将被删除：{Style.RESET_ALL}")
            for file in obsolete_files_found:
                print(f"  {file}")
        all_files_to_delete = files_to_delete + obsolete_files_found  # 新增合并逻辑

        print(f"{Fore.CYAN}△可清理的磁盘空间: {(total_size + obsolete_total) / (1024 * 1024):.2f} MB{Style.RESET_ALL}")
        print(f"{Fore.GREEN}△是否确认删除这些文件？(y/n): {Style.RESET_ALL}", flush=True)
        confirm = input()
        if confirm.lower() == 'y':
            success_count = 0
            for file_path in all_files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除: {file_path}{Style.RESET_ALL}")
                    success_count += 1
                except Exception as e:
                    print(f"{Fore.RED}×删除失败[{file_path}]: {str(e)}{Style.RESET_ALL}")
            print(f"操作完成！成功删除 {success_count}/{len(all_files_to_delete)} 个文件")
        else:
            print(f"{Fore.RED}△删除操作已取消{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的临时/损坏文件<<<")

def delete_specific_image_files():
    """
    从相对路径查找并删除所有 .png、.webp 和 .jpg/jpeg 文件，排除 welcome.png。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    users_dir = find_users_dir(script_dir)
    target_dir = os.path.join(users_dir, "guest_user", "comfyd_inputs")
    if not os.path.exists(target_dir):
        print(f"{Fore.RED}△未找到指定目录: {target_dir}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}△正在清理目录 '{target_dir}' 中的临时图片缓存...{Style.RESET_ALL}")

    total_size = 0
    files_found = False
    files_to_delete = []

    for root, _, files in os.walk(target_dir):
        for file in files:
            if (file.endswith(".png") or file.endswith(".webp") or file.endswith(".jpg") or file.endswith(".jpeg")) and file != "welcome.png":
                files_found = True
                file_path = os.path.join(root, file)
                files_to_delete.append(file_path)
                total_size += os.path.getsize(file_path)
    if files_found:
        print(f"{Fore.YELLOW}△以下临时图片缓存文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")
        print(f"{Fore.CYAN}△可清理的磁盘空间: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")
        print(f"{Fore.GREEN}△是否确认删除这些文件？(y/n): {Style.RESET_ALL}", flush=True)
        confirm = input()
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}△删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的临时图片缓存<<<")
        print()

def delete_log_files():
    """
    删除与脚本所在位置一致的 logs 目录下的所有 .logs 文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")

    if not os.path.exists(logs_dir):
        print(f"{Fore.RED}△未找到指定日志目录: {logs_dir}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}△正在清理目录 '{logs_dir}' 中的日志文件...{Style.RESET_ALL}")

    total_size = 0
    files_found = False
    files_to_delete = []

    for root, _, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".log"):
                files_found = True
                file_path = os.path.join(root, file)
                files_to_delete.append(file_path)
                total_size += os.path.getsize(file_path)

    if files_found:
        print(f"{Fore.YELLOW}△以下日志文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")

        print(f"{Fore.CYAN}△可清理的磁盘空间: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")
        print(f"{Fore.GREEN}△是否确认删除这些日志文件？(y/n): {Style.RESET_ALL}", flush=True)
        confirm = input()
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除日志文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}△删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的日志文件<<<")
        print()

def download_file_with_resume(link, file_path, position, result_queue, max_retries=5, lock=None):
    partial_file_path = file_path + ".partial"
    retries = 0
    while retries < max_retries:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(partial_file_path):
                resume_size = os.path.getsize(partial_file_path)
                headers = {'Range': f"bytes={resume_size}-"}
            else:
                resume_size = 0
                headers = {}

            response = requests.get(link, stream=True, headers=headers, timeout=(30, 60))

            mode = 'ab'
            if response.status_code == 200:
                resume_size = 0
                mode = 'wb'
                total_size = int(response.headers.get('content-length', 0))
            elif response.status_code == 206:
                # 服务器支持断点续传
                content_range = response.headers.get('content-range', '')
                match = re.search(r'/(\d+)$', content_range)
                if match:
                    total_size = int(match.group(1))
                else:
                    total_size = int(response.headers.get('content-length', 0)) + resume_size
            else:

                total_size = int(response.headers.get('content-length', 0)) + resume_size

            block_size = 8192

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(partial_file_path, mode) as file, tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    position=position,
                    initial=resume_size,
                    dynamic_ncols=True,
                    leave=False,  # 任务完成后清除进度条，避免视觉残留
                    file=sys.stdout,
                    miniters=1,
                    mininterval=0.1,  # 增加更新间隔，减少性能影响
                    disable=False
            ) as progress_bar:
                start_time = time.time()
                last_update_time = start_time

                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

                    # 添加超时检测：如果超过60秒没有数据，认为连接已断
                    current_time = time.time()
                    if current_time - last_update_time > 60:
                        raise requests.exceptions.Timeout("下载超时，超过60秒没有数据")
                    last_update_time = current_time

                file.flush()
                os.fsync(file.fileno())

                # 校验文件大小
                downloaded_size = os.path.getsize(partial_file_path)
                if downloaded_size != total_size:
                    raise requests.exceptions.RequestException(f"文件大小校验失败：预期 {total_size} 字节，实际 {downloaded_size} 字节")

            final_file_path = os.path.normpath(file_path)
            partial_file_path = os.path.normpath(partial_file_path)
            os.rename(partial_file_path, final_file_path)

            try:
                official_sha256 = get_modelscope_file_sha256(link, verbose=False)
                if official_sha256:
                    local_sha256 = calculate_sha256(final_file_path)
                    if local_sha256.lower() != official_sha256.lower():
                        os.remove(final_file_path)
                        raise requests.exceptions.RequestException(f"SHA256校验失败 (预期: {official_sha256}, 实际: {local_sha256})")
            except Exception as e:
                if "SHA256校验失败" in str(e):
                    raise e

            tqdm.write(f"{Fore.GREEN}√下载完成：{final_file_path}{Style.RESET_ALL}")

            if lock:
                with lock:
                    remove_link_from_downloadlist(link)

            result_queue.put(True)
            return

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            tqdm.write(f"{Fore.RED}△下载失败，正在重试... 错误：{e}{Style.RESET_ALL}")
            retries += 1
            time.sleep(5)
        except Exception as e:
            tqdm.write(f"{Fore.RED}发生错误：{e}{Style.RESET_ALL}")
            result_queue.put(False)
            return

    tqdm.write(f"△下载链接失败：{link}")
    result_queue.put(False)

def remove_link_from_downloadlist(link):
    """
    删除下载列表中已成功下载的条目
    :param link: 下载链接
    :return: None
    """
    with open("downloadlist.txt", "r") as f:
        lines = f.readlines()

    with open("downloadlist.txt", "w") as f:
        for line in lines:
            if link.strip() not in line.strip():
                f.write(line)
def trigger_manual_download():
    """手动触发指定文件下载"""
    path_mapping = load_model_paths()

    for link in MANUAL_DOWNLOAD_LIST:
        if "SimpleModels/" in link:
            path_part = link.split("SimpleModels/", 1)[1]
            path_parts = path_part.split('/')
            path_type = path_parts[0]
            rel_path = '/'.join(path_parts[1:])
        else:
            continue

        sorted_base_dir = sorted(
            path_mapping.get(path_type, []),
            key=lambda x: (
                0 if "SimpleModels" in x else
                1 if any(part == "models" for part in x.split(os.sep)) else 2,
                x
            )
        )

        target_base_dir = None
        for base_dir in sorted_base_dir:
            if os.path.exists(base_dir):
                target_base_dir = base_dir
                break
        if not target_base_dir:
            continue

        file_name = os.path.basename(link)
        save_path = os.path.join(target_base_dir, rel_path)

        if os.path.exists(save_path):
            print(f"{Fore.GREEN}△文件已存在，跳过下载: {save_path}{Style.RESET_ALL}")
            continue

        print(f"{Fore.CYAN}△开始下载: {file_name}{Style.RESET_ALL}")
        result_queue = queue.Queue()
        download_file_with_resume(link, save_path, 0, result_queue)

def auto_download_missing_files_with_retry(max_threads=5):
    if not os.path.exists("downloadlist.txt"):
        print("未找到 'downloadlist.txt' 文件。")
        return

    with open("downloadlist.txt", "r") as f:
        links = f.readlines()

    if not links:
        print("没有缺失文件需要下载！")
        return

    path_mapping = load_model_paths()
    result_queue = queue.Queue()
    lock = threading.Lock()

    task_queue = queue.Queue()
    # position 不再直接使用索引，而是通过 Slot 机制动态分配
    for index, line in enumerate(links):
        task_queue.put(line.strip())

    # 创建可用 Slot 队列，限制同时显示的进度条数量等于线程数
    position_slots = queue.Queue()
    for i in range(max_threads):
        position_slots.put(i)

    def worker():
        while not task_queue.empty():
            try:
                line = task_queue.get_nowait()
                
                # 获取一个可用的 Slot 用于显示进度条
                try:
                    position = position_slots.get(timeout=30)
                except queue.Empty:
                    # 理论上不应发生，但作为防守
                    position = 0 
                
                link, size = line.split(',')
                size_mb = int(size) / (1024 * 1024)
                
                # 使用 tqdm.write 避免破坏进度条
                tqdm.write(f"{Fore.CYAN}▶ 正在下载: {link} ({size_mb:.1f}MB){Style.RESET_ALL}")

                # 检查是否是原始仓库链接
                if link.startswith(CURRENT_DOWNLOAD_PREFIX):
                    relative_path = link.replace(CURRENT_DOWNLOAD_PREFIX, "", 1).strip()
                    relative_path_without_prefix = relative_path.replace("SimpleModels/", "", 1)
                    path_type = relative_path_without_prefix.split('/')[0]
                else:
                    found_path = None
                    for package_name, package_info in packages.items():
                        for file_path, _ in package_info["files"]:
                            if link in file_path:
                                found_path = file_path
                                break
                        if found_path:
                            break

                    if found_path:
                        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
                        url_match = re.search(url_pattern, found_path)
                        if url_match:
                            save_directory = found_path.split(url_match.group(0))[0].rstrip('/')
                            path_type = save_directory.split('/')[0] if save_directory else "default"
                            url_path = url_match.group(0)
                            file_name_from_url = os.path.basename(url_path)
                            relative_path = os.path.join(save_directory, file_name_from_url)
                            relative_path_without_prefix = relative_path.replace("SimpleModels/", "", 1) if "SimpleModels/" in relative_path else relative_path
                        else:
                            path_type = found_path.split('/')[0].lower()
                            relative_path = found_path
                            relative_path_without_prefix = relative_path.replace("SimpleModels/", "", 1) if "SimpleModels/" in relative_path else relative_path
                    else:
                        url_parts = link.split('/')
                        possible_types = ["checkpoints", "loras", "controlnet", "embeddings", "vae", "inpaint", "ipadapter"]
                        path_type = "default"
                        for part in url_parts:
                            if part.lower() in possible_types:
                                path_type = part.lower()
                                break
                        file_name_from_url = os.path.basename(link)
                        relative_path = os.path.join(path_type, file_name_from_url)
                        relative_path_without_prefix = relative_path

                sorted_base_dir = sorted(
                    path_mapping.get(path_type, []),
                    key=lambda x: (
                        0 if "SimpleModels" in x else
                        1 if any(part == "models" for part in x.split(os.sep)) else
                        2,
                        x
                    )
                )

                target_base_dir = None
                for base_dir in sorted_base_dir:
                    if os.path.exists(base_dir):
                        target_base_dir = base_dir
                        break
                if not target_base_dir:
                    target_base_dir = sorted_base_dir[0]
                    try:
                        os.makedirs(target_base_dir, exist_ok=True)
                        tqdm.write(f"{Fore.YELLOW}△自动创建缺失目录: {target_base_dir}{Style.RESET_ALL}")
                    except Exception as e:
                        tqdm.write(f"{Fore.RED}×目录创建失败[{target_base_dir}]: {str(e)}{Style.RESET_ALL}")
                        # 失败也要归还 slot
                        position_slots.put(position)
                        task_queue.task_done()
                        continue

                file_name = os.path.basename(relative_path)
                file_sub_dir = os.path.dirname(relative_path_without_prefix).replace(path_type, "").strip('/')
                save_dir = os.path.join(target_base_dir, file_sub_dir)
                file_path = os.path.join(save_dir, file_name)
                download_file_with_resume(link, file_path, position, result_queue, 5, lock)

                # 下载完成（无论成功失败），归还 Slot
                position_slots.put(position)
                task_queue.task_done()
            except queue.Empty:
                break

    threads = []
    for _ in range(max_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    success_count = 0
    fail_count = 0

    while not result_queue.empty():
        success = result_queue.get()
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"√下载成功：{success_count}个")
    print(f"×下载失败：{fail_count}个")

    if fail_count == 0 and success_count > 0:
        if os.path.exists("downloadlist.txt"):
            os.remove("downloadlist.txt")
            print("√下载完成，执行重新检测")
            validate_files(packages)
    else:
        print(f"△有{fail_count}个文件下载失败，请检查网络连接或手动下载文件。")

def get_download_links_for_package(packages, download_list_path):
    """
    根据 packages 中的 files 列表生成路径，并与 downloadlist.txt 中的需求进行比对，
    更新 downloadlist.txt 中需要下载的文件，只保留 files 中有的文件链接。
    """
    if not os.path.exists(download_list_path):
        print(f"{Fore.RED}>>>downloadlist.txt不存在，输入【R】重新检测<<<{Style.RESET_ALL}")
        return []

    with open(download_list_path, "r") as f:
        existing_links = [line.strip().split(",")[0] for line in f.readlines()]

    valid_files = []
    added_links = set()
    with open(download_list_path, "r") as f:
        existing_lines = [line.strip() for line in f.readlines()]

    for line in existing_lines:
        existing_link = line.split(",")[0]
        for package_name, package_info in packages.items():
            for full_file_path, file_size in package_info["files"]:
                url_pattern = r'https?://[^\s/$.?#].[^\s]*'
                url_match = re.search(url_pattern, full_file_path)

                if url_match:
                    generated_link = url_match.group(0)
                    save_directory = full_file_path.split(url_match.group(0))[0].rstrip('/')
                    file_name = os.path.basename(url_match.group(0))
                else:
                    generated_link = f"{CURRENT_DOWNLOAD_PREFIX}SimpleModels/{full_file_path}"

                if generated_link == existing_link and generated_link not in added_links:
                    valid_files.append((generated_link, file_size))
                    added_links.add(generated_link)
                    break

    valid_files = sorted(valid_files, key=lambda x: x[1])

    with open(download_list_path, "w") as f:
        for link, size in valid_files:
            f.write(f"{link},{size}\n")

    print(f"{Fore.YELLOW}>>>下载列表已更新，开始下载（关闭窗口可中断）。<<<{Style.RESET_ALL}")

    return valid_files

def delete_package(package_name, packages):
    """删除指定模型包文件（基于config路径配置）"""
    try:
        path_mapping = load_model_paths()
    except Exception as e:
        print(f"{Fore.RED}× 路径配置加载失败: {str(e)}{Style.RESET_ALL}")
        return

    if package_name not in packages:
        print(f"{Fore.RED}× 无效的模型包名称！{Style.RESET_ALL}")
        return

    package = packages[package_name]
    print(f"\n{Fore.CYAN}△ 开始处理模型包：{package['name']}{Style.RESET_ALL}")

    file_refs = defaultdict(list)
    for pkg_name, pkg_info in packages.items():
        for file_entry in pkg_info["files"]:
            path_with_url = file_entry[0]
            url_pattern = r'https?://[^\s/$.?#].[^\s]*'
            url_match = re.search(url_pattern, path_with_url)

            if url_match:
                url = url_match.group(0)
                path_part = path_with_url.split(url)[0].rstrip('/')
                file_name = os.path.basename(url)

                if path_part:
                    path_parts = path_part.split('/')
                    file_type = path_parts[0] if path_parts else ""
                    rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""

                    for base_dir in path_mapping.get(file_type, []):
                        if rel_path:
                            full_path = os.path.join(base_dir, rel_path, file_name)
                        else:
                            full_path = os.path.join(base_dir, file_name)
                        if os.path.exists(full_path):
                            file_refs[full_path].append(pkg_name)
            else:
                path_parts = path_with_url.split('/')
                if len(path_parts) < 1: continue

                file_type = path_parts[0]
                rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""

                for base_dir in path_mapping.get(file_type, []):
                    full_path = os.path.join(base_dir, rel_path)
                    if os.path.exists(full_path):
                        file_refs[full_path].append(pkg_name)

    delete_candidates = []
    shared_files = []

    for file_entry in package["files"]:
        path_with_url = file_entry[0]
        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        url_match = re.search(url_pattern, path_with_url)

        if url_match:
            url = url_match.group(0)
            path_part = path_with_url.split(url)[0].rstrip('/')
            file_name = os.path.basename(url)
            found = False

            if path_part:
                path_parts = path_part.split('/')
                file_type = path_parts[0] if path_parts else ""
                rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""

                for base_dir in path_mapping.get(file_type, []):
                    if rel_path:
                        full_path = os.path.join(base_dir, rel_path, file_name)
                    else:
                        full_path = os.path.join(base_dir, file_name)
                    if os.path.exists(full_path):
                        if len(file_refs[full_path]) == 1 and file_refs[full_path][0] == package_name:
                            delete_candidates.append(full_path)
                        else:
                            shared_files.append(full_path)
                        found = True
                        break

            if not found:
                if path_part:
                    path_parts = path_part.split('/')
                    if path_parts:
                        file_type = path_parts[0]
                        rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""
                        default_base_dir = os.path.join(simplemodels_root, file_type)
                        if rel_path:
                            default_path = os.path.join(default_base_dir, rel_path, file_name)
                        else:
                            default_path = os.path.join(default_base_dir, file_name)
                    else:
                        default_base_dir = simplemodels_root
                        default_path = os.path.join(default_base_dir, file_name)
                else:
                    default_base_dir = simplemodels_root
                    default_path = os.path.join(default_base_dir, file_name)

                if os.path.exists(default_path):
                    if len(file_refs[default_path]) == 1 and file_refs[default_path][0] == package_name:
                        delete_candidates.append(default_path)
                    else:
                        shared_files.append(default_path)
        else:
            path_parts = path_with_url.split('/')
            if len(path_parts) < 1: continue

            file_type = path_parts[0]
            rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""
            found = False

            for base_dir in path_mapping.get(file_type, []):
                full_path = os.path.join(base_dir, rel_path)
                if os.path.exists(full_path):
                    if len(file_refs[full_path]) == 1 and file_refs[full_path][0] == package_name:
                        delete_candidates.append(full_path)
                    else:
                        shared_files.append(full_path)
                    found = True
                    break

            if not found:
                default_path = os.path.join(simplemodels_root, file_type, rel_path)
                if os.path.exists(default_path):
                    if len(file_refs[default_path]) == 1 and file_refs[default_path][0] == package_name:
                        delete_candidates.append(default_path)
                    else:
                        shared_files.append(default_path)

    if shared_files:
        print(f"\n{Fore.YELLOW}△ 以下文件被其他模型包共享：{Style.RESET_ALL}")
        for path in shared_files:
            print(f"  {path}")

    if delete_candidates:
        print(f"\n{Fore.YELLOW}△ 以下孤立文件将被删除：{Style.RESET_ALL}")
        total_size = 0
        for path in delete_candidates:
            try:
                size = os.path.getsize(path)
                print(f"  {path} ({size/1024/1024:.1f}MB)")
                total_size += size
            except:
                print(f"  {path} (大小未知)")

        print(f"{Fore.CYAN}△ 总计释放空间: {total_size/1024/1024/1024:.2f}GB{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}是否确认删除？(y/n): {Style.RESET_ALL}", flush=True)
        confirm = input()
        if confirm.lower() == 'y':
            success = 0
            for path in delete_candidates:
                try:
                    os.remove(path)
                    print(f"{Fore.GREEN}✓ 已删除: {path}{Style.RESET_ALL}")
                    success += 1
                except Exception as e:
                    print(f"{Fore.RED}× 删除失败: {path} ({str(e)}){Style.RESET_ALL}")

            print(f"\n{Fore.GREEN}✓ 模型包{selected_package['name']}孤立文件已清除{Style.RESET_ALL}")
            validate_files(packages)
        else:
            print(f"{Fore.BLUE}× 操作已取消{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}△ 未找到可安全删除的文件{Style.RESET_ALL}")

def delete_package_force(package_name, packages):
    """强制删除指定模型包文件（不检查关联性）"""
    try:
        path_mapping = load_model_paths()
    except Exception as e:
        print(f"{Fore.RED}× 路径配置加载失败: {str(e)}{Style.RESET_ALL}")
        return

    if package_name not in packages:
        print(f"{Fore.RED}× 无效的模型包名称！{Style.RESET_ALL}")
        return

    package = packages[package_name]
    print(f"\n{Fore.RED}!!! 正在执行强制删除操作 !!!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}△ 开始处理模型包：{package['name']}{Style.RESET_ALL}")

    delete_candidates = []

    for file_entry in package["files"]:
        path_with_url = file_entry[0]
        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        url_match = re.search(url_pattern, path_with_url)

        if url_match:
            url = url_match.group(0)
            path_part = path_with_url.split(url)[0].rstrip('/')
            file_name = os.path.basename(url)
            found = False

            if path_part:
                path_parts = path_part.split('/')
                file_type = path_parts[0] if path_parts else ""
                rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""

                for base_dir in path_mapping.get(file_type, []):
                    if rel_path:
                        full_path = os.path.join(base_dir, rel_path, file_name)
                    else:
                        full_path = os.path.join(base_dir, file_name)
                    if os.path.exists(full_path):
                        delete_candidates.append(full_path)
                        found = True
                        break

            if not found:
                if path_part:
                    path_parts = path_part.split('/')
                    if path_parts:
                        file_type = path_parts[0]
                        rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""
                        default_base_dir = os.path.join(simplemodels_root, file_type)
                        if rel_path:
                            default_path = os.path.join(default_base_dir, rel_path, file_name)
                        else:
                            default_path = os.path.join(default_base_dir, file_name)
                    else:
                        default_base_dir = simplemodels_root
                        default_path = os.path.join(default_base_dir, file_name)
                else:
                    default_base_dir = simplemodels_root
                    default_path = os.path.join(default_base_dir, file_name)

                if os.path.exists(default_path):
                    delete_candidates.append(default_path)
        else:
            path_parts = path_with_url.split('/')
            if len(path_parts) < 1: continue

            file_type = path_parts[0]
            rel_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ""
            found = False

            for base_dir in path_mapping.get(file_type, []):
                full_path = os.path.join(base_dir, rel_path)
                if os.path.exists(full_path):
                    delete_candidates.append(full_path)
                    found = True
                    break

            if not found:
                default_path = os.path.join(simplemodels_root, file_type, rel_path)
                if os.path.exists(default_path):
                    delete_candidates.append(default_path)

    if delete_candidates:
        print(f"\n{Fore.RED}△ 以下文件将被【强制删除】（不检查其他模型包依赖）：{Style.RESET_ALL}")
        total_size = 0
        for path in delete_candidates:
            try:
                size = os.path.getsize(path)
                print(f"  {path} ({size/1024/1024:.1f}MB)")
                total_size += size
            except:
                print(f"  {path} (大小未知)")

        print(f"{Fore.CYAN}△ 总计释放空间: {total_size/1024/1024/1024:.2f}GB{Style.RESET_ALL}")

        print(f"\n{Fore.RED}此操作不可逆且可能破坏其他模型包完整性，是否确认强制删除？(输入 'force' 确认): {Style.RESET_ALL}", flush=True)
        confirm = input().lower()
        if confirm == 'force':
            success = 0
            for path in delete_candidates:
                try:
                    os.remove(path)
                    print(f"{Fore.GREEN}✓ 已删除: {path}{Style.RESET_ALL}")
                    success += 1
                except Exception as e:
                    print(f"{Fore.RED}× 删除失败: {path} ({str(e)}){Style.RESET_ALL}")

            print(f"\n{Fore.GREEN}✓ 模型包{package['name']}文件已强制清除{Style.RESET_ALL}")
            validate_files(packages)
        else:
            print(f"{Fore.BLUE}× 操作已取消{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}△ 未找到该模型包的任何文件{Style.RESET_ALL}")

def get_gpu_arch_str():
    """获取GPU架构字符串，如sm120等"""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch_str = f"sm{major}{minor}"
            return arch_str
        else:
            return "cpu"
    except Exception as e:
        print(f"获取GPU架构失败: {e}")
        return "cpu"

def filter_packages_by_gpu_arch(packages):
    """
    根据GPU架构过滤package
    - 当sm120时，只显示带fp4的package和其他无标识package
    - 当不等于sm120且高于10系显卡时，只显示带int4的package和其他无标识package
    - 对于10系及以下显卡，只显示无标识package
    - 当GPU小于等于20系时，不显示"[29]双截棍fp4-QwenEdit+图像编辑"和"[28]双截棍int4-QwenEdit+图像编辑"
    """
    # 获取GPU架构
    gpu_arch = get_gpu_arch_str()
    filtered_packages = {}

    is_legacy_gpu = False
    is_20_series_or_lower = False
    if gpu_arch.startswith('sm'):
        try:
            arch_number = int(gpu_arch[2:])
            # 计算能力<=61的视为10系及以下显卡
            is_legacy_gpu = arch_number <= 61
            # 计算能力<=75的视为20系及以下显卡（RTX 20系列计算能力为7.5）
            is_20_series_or_lower = arch_number <= 75
        except ValueError:
            pass

    for package_key, package_info in packages.items():
        package_name = package_info["name"]

        # 当GPU小于等于20系时，不显示这两个特定的package
        if is_20_series_or_lower:
            if package_name == "[29]双截棍fp4-QwenEdit+图像编辑" or package_name == "[28]双截棍int4-QwenEdit+图像编辑":
                continue

        has_int4 = 'int4' in package_name.lower()
        has_fp4 = 'fp4' in package_name.lower()

        # 10系及以下显卡特殊处理：只保留无标识package
        if is_legacy_gpu:
            if not has_int4 and not has_fp4:
                filtered_packages[package_key] = package_info
        else:
            if has_int4 and has_fp4:
                continue

            if gpu_arch == 'sm120':
                if has_fp4 or (not has_int4 and not has_fp4):
                    filtered_packages[package_key] = package_info
            else:
                if has_int4 or (not has_int4 and not has_fp4):
                    filtered_packages[package_key] = package_info

    return filtered_packages

packages = {
    "base_package": {
        "id": 1,
        "name": "[1]基础模型包[Z-image-Turbo]",
        "note": "Z-image-Turbo-默认模型[Z-image-Turbo-fp16]|显存需求：★★☆ 速度：★★★",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/VerStella/z_image_turbo_comfyui/resolve/master/split_files/diffusion_models/z_image_turbo_bf16.safetensors", 12309866400),
            ("text_encoders/https://www.modelscope.cn/models/VerStella/z_image_turbo_comfyui/resolve/master/split_files/text_encoders/qwen_3_4b.safetensors", 8044982048),
            ("model_patches/https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/master/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors", 6712485600),
            ("vae/ae.safetensors", 335304388),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("clip_vision/clip_vision_vit_h.safetensors", 1972298538),
            ("clip_vision/model_base_caption_capfilt_large.pth", 896081425),
            ("clip_vision/https://www.modelscope.cn/models/windecay/WD-tagger/resolve/master/wd-eva02-large-tagger-v3.onnx", 1260435999),
            ("clip_vision/https://www.modelscope.cn/models/windecay/WD-tagger/resolve/master/wd-eva02-large-tagger-v3.csv", 308468),
            ("clip_vision/clip-vit-large-patch14/merges.txt", 524619),
            ("clip_vision/clip-vit-large-patch14/special_tokens_map.json", 389),
            ("clip_vision/clip-vit-large-patch14/tokenizer_config.json", 905),
            ("clip_vision/clip-vit-large-patch14/vocab.json", 961143),
            ("configs/anything_v3.yaml", 1933),
            ("configs/v1-inference.yaml", 1873),
            ("configs/v1-inference_clip_skip_2.yaml", 1933),
            ("configs/v1-inference_clip_skip_2_fp16.yaml", 1956),
            ("configs/v1-inference_fp16.yaml", 1896),
            ("configs/v1-inpainting-inference.yaml", 1992),
            ("configs/v2-inference-v.yaml", 1815),
            ("configs/v2-inference-v_fp32.yaml", 1816),
            ("configs/v2-inference.yaml", 1789),
            ("configs/v2-inference_fp32.yaml", 1790),
            ("configs/v2-inpainting-inference.yaml", 4450),
            ("controlnet/detection_Resnet50_Final.pth", 109497761),
            ("controlnet/fooocus_ip_negative.safetensors", 65616),
            ("controlnet/parsing_parsenet.pth", 85331193),
            ("controlnet/lllyasviel/Annotators/body_pose_model.pth", 209267595),
            ("controlnet/lllyasviel/Annotators/facenet.pth", 153718792),
            ("controlnet/lllyasviel/Annotators/hand_pose_model.pth", 147341049),
            ("controlnet/hr16/DWPose-TorchScript-BatchSize5/https://www.modelscope.cn/models/svjack/DWPose-TorchScript-BatchSize5/resolve/master/dw-ll_ucoco_384_bs5.torchscript.pt", 135059124),
            ("controlnet/yzd-v/DWPose/https://www.modelscope.cn/models/zhangjin/DWPose/resolve/master/yolox_l.onnx", 216746733),
            ("inpaint/fooocus_inpaint_head.pth", 52602),
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/isnet-anime.onnx", 176069933),
            ("inpaint/isnet-general-use.onnx", 178648008),
            ("inpaint/sam_vit_b_01ec64.pth", 375042383),
            ("inpaint/sam_vit_l_0b3195.pth", 1249524607),
            ("inpaint/silueta.onnx", 44173029),
            ("inpaint/u2net.onnx", 175997641),
            ("inpaint/u2netp.onnx", 4574861),
            ("inpaint/u2net_cloth_seg.onnx", 176194565),
            ("inpaint/u2net_human_seg.onnx", 175997641),
            ("llms/bert-base-uncased/config.json", 570),
            ("llms/bert-base-uncased/model.safetensors", 440449768),
            ("llms/bert-base-uncased/tokenizer.json", 466062),
            ("llms/bert-base-uncased/tokenizer_config.json", 28),
            ("llms/bert-base-uncased/vocab.txt", 231508),
            ("llms/Helsinki-NLP/opus-mt-zh-en/config.json", 1394),
            ("llms/Helsinki-NLP/opus-mt-zh-en/generation_config.json", 293),
            ("llms/Helsinki-NLP/opus-mt-zh-en/metadata.json", 1477),
            ("llms/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin", 312087009),
            ("llms/Helsinki-NLP/opus-mt-zh-en/source.spm", 804677),
            ("llms/Helsinki-NLP/opus-mt-zh-en/target.spm", 806530),
            ("llms/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json", 44),
            ("llms/Helsinki-NLP/opus-mt-zh-en/vocab.json", 1617902),
            ("llms/superprompt-v1/config.json", 1512),
            ("llms/superprompt-v1/generation_config.json", 142),
            ("llms/superprompt-v1/model.safetensors", 307867048),
            ("llms/superprompt-v1/README.md", 3661),
            ("llms/superprompt-v1/spiece.model", 791656),
            ("llms/superprompt-v1/tokenizer.json", 2424064),
            ("llms/superprompt-v1/tokenizer_config.json", 2539),
            ("rembg/RMBG-1.4.pth", 176718373),
            ("vae_approx/vaeapp_sd15.pth", 213777),
            ("vae_approx/xl-to-v1_interposer-v4.0.safetensors", 5667280),
            ("vae_approx/xlvaeapp.pth", 213777),
            ("clip/clip_l.safetensors", 246144152),
            ("vae/ponyDiffusionV6XL_vae.safetensors", 334641162),
            ("ultralytics/bbox/https://www.modelscope.cn/models/ACCC1380/Adetailer_model/resolve/master/face_yolov8m.pt", 52026019),
            ("ultralytics/bbox/https://www.modelscope.cn/models/ACCC1380/Adetailer_model/resolve/master/hand_yolov8s.pt", 22507643),
        ],
        "download_links": []
    },
    "extension_package": {
        "id": 2,
        "name": "[2]IC-Light重打光",
        "note": "IC-Light图像重打光预置包|显存需求：★★ 速度：★★★☆",
        "files": [
            ("checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("unet/iclight_sd15_fbc_unet_ldm.safetensors", 1719167896),
            ("unet/iclight_sd15_fc_unet_ldm.safetensors", 1719144856),
        ],
        "download_links": []
    },
        "Flux_aio_package": {
        "id": 3,
        "name": "[3]Flux_AIO扩展包",
        "note": "Flux全功能-默认模型[Flux_Q5K_M]|显存需求：★★★☆ 速度：★★",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/flux1-fill-dev-OneReward_fp8.safetensors", 11902532704),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": []
    },
        "SD15_aio_package": {
        "id": 4,
        "name": "[4]SD1.5_AIO扩展包",
        "note": "SD1.5全功能-默认模型[realisticVision]|显存需求：★ 速度：★★★★",
        "files": [
            ("checkpoints/realisticVisionV60B1_v51VAE.safetensors", 2132625894),
            ("clip/sd15_clip_model.fp16.safetensors", 246144864),
            ("controlnet/control_v11f1e_sd15_tile_fp16.safetensors", 722601104),
            ("controlnet/control_v11f1p_sd15_depth_fp16.safetensors", 722601100),
            ("controlnet/control_v11p_sd15_canny_fp16.safetensors", 722601100),
            ("controlnet/control_v11p_sd15_openpose_fp16.safetensors", 722601100),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("inpaint/sd15_powerpaint_brushnet_clip_v2_1.bin", 492401329),
            ("inpaint/sd15_powerpaint_brushnet_v2_1.safetensors", 3544366408),
            ("insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("insightface/models/buffalo_l/w600k_r50.onnx", 174383860),
            ("ipadapter/ip-adapter-faceid-plusv2_sd15.bin", 156558509),
            ("ipadapter/ip-adapter_sd15.safetensors", 44642768),
            ("loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors", 51059544),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604)
        ],
        "download_links": []
    },
    "one_key_pose_package": {
        "id": 5,
        "name": "[5]一键Pose骨骼图预置包",
        "note": "使用SDPose、DWpose进行预处理姿势|显存需求：★★ 速度：★★★★★",
        "files": [
            ("controlnet/hr16/DWPose-TorchScript-BatchSize5/https://www.modelscope.cn/models/svjack/DWPose-TorchScript-BatchSize5/resolve/master/dw-ll_ucoco_384_bs5.torchscript.pt", 135059124),
            ("controlnet/yzd-v/DWPose/https://www.modelscope.cn/models/zhangjin/DWPose/resolve/master/yolox_l.onnx", 216746733),
            ("SDPose_OOD/SDPose-Wholebody/vae/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/vae/config.json", 611),
            ("SDPose_OOD/SDPose-Wholebody/vae/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/vae/diffusion_pytorch_model.safetensors", 334643276),
            ("SDPose_OOD/SDPose-Wholebody/unet/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/unet/config.json", 1859),
            ("SDPose_OOD/SDPose-Wholebody/unet/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/unet/diffusion_pytorch_model.safetensors", 3470311272),
            ("SDPose_OOD/SDPose-Wholebody/tokenizer/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/tokenizer/merges.txt", 524619),
            ("SDPose_OOD/SDPose-Wholebody/tokenizer/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/tokenizer/special_tokens_map.json", 460),
            ("SDPose_OOD/SDPose-Wholebody/tokenizer/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/tokenizer/tokenizer_config.json", 824),
            ("SDPose_OOD/SDPose-Wholebody/tokenizer/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/tokenizer/vocab.json", 1059962),
            ("SDPose_OOD/SDPose-Wholebody/decoder/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/decoder/decoder.safetensors", 28196828),
            ("SDPose_OOD/SDPose-Wholebody/scheduler/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/scheduler/scheduler_config.json", 344),
            ("SDPose_OOD/SDPose-Wholebody/text_encoder/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/text_encoder/model.safetensors", 1361597018),
            ("SDPose_OOD/SDPose-Wholebody/text_encoder/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/text_encoder/config.json", 633),
            ("yolo/https://www.modelscope.cn/models/Sunjian520/SDPose-Wholebody/resolve/master/yolo11x.pt", 114636239),
        ],
        "download_links": []
    },
        "Qwen3_package": {
        "id": 6,
        "name": "[6]Qwen3-VL-4B反推扩展包",
        "note": "本地多模态大语言模型[反推、翻译、扩写]|显存需求：★★ 速度：★★",
        "files": [
                ("LLM/Qwen3-VL-4B-Instruct-abliterated/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-4B-Instruct-abliterated/Qwen3-VL-4B-Instruct-abliterated-v1.Q8_0.gguf",4280407104),
                ("LLM/Qwen3-VL-4B-Instruct-abliterated/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/LLM/Qwen3-VL-4B-Instruct-abliterated/Qwen3-VL-4B-Instruct-abliterated-v1.mmproj-Q8_0.gguf",453974752)
        ],
        "download_links": []
    },
        "x1-okremovebg_package": {
        "id": 7,
        "name": "[7]一键抠图",
        "note": "抠图去背景神器|显存需求：★ 速度：★★★★★",
        "files": [
            ("rembg/ckpt_base.pth", 367520613),
            ("rembg/RMBG-1.4.pth", 176718373),
            ("rembg/General.safetensors", 884878856),
            ("rembg/Portrait.safetensors", 884878856)
        ],
        "download_links": []
    },
        "x2-okimagerepair_package": {
        "id": 8,
        "name": "[8]一键修复",
        "note": "上色、修复模糊、旧照片[XL/Flux]|显存需求：★★★ 速度：★☆",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/ae.safetensors", 335304388),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/detection_Resnet50_Final.pth", 109497761),
            ("controlnet/facerestore_models/codeformer-v0.1.0.pth", 376637898),
            ("controlnet/facerestore_models/GFPGANv1.4.pth", 348632874),
            ("clip_vision/clip_vision_vit_h.safetensors", 1972298538),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("upscale_models/4xNomos8kSCHAT-L.pth", 331564661)
        ],
        "download_links": []
    },
        "x3-swapface_package": {
        "id": 9,
        "name": "[9]一键换脸",
        "note": "高精度换脸-默认模型[OneReward_fp8]|显存需求：★★★ 速度：★★",
        "files": [
            ("checkpoints/flux1-fill-dev-OneReward_fp8.safetensors", 11902532704),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("loras/flux1-turbo.safetensors", 694082424),
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/https://modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("style_models/flux1-redux-dev.safetensors", 129063232),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("loras/comfyui_portrait_lora64.safetensors",612742344),
            ("controlnet/detection_Resnet50_Final.pth", 109497761),
            ("controlnet/parsing_bisenet.pth", 53289463),
        ],
        "download_links": []
    },
        "Flux_aio_plus_package": {
        "id": 10,
        "name": "[10]Flux_AIO_plus扩展包",
        "note": "Flux全功能-默认模型[Fluxdev_fp8]|显存需求：★★★★ 速度：★★☆",
        "files": [
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("checkpoints/flux1-dev-fp8.safetensors", 11901525888),
            ("checkpoints/flux1-fill-dev-OneReward_fp8.safetensors", 11902532704),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": [
        "【选配】https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/flux1-dev-fp8.safetensors"
        ]
    },
        "clothing_plus_package": {
        "id": 11,
        "name": "[11]换装plus包",
        "note": "万物迁移-默认模型[Fluxdev_fp8]|显存需求：★★★☆ 速度：★★★",
        "files": [
            ("inpaint/groundingdino_swint_ogc.pth", 693997677),
            ("inpaint/https://modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/inpaint/GroundingDINO_SwinT_OGC.cfg.py", 1006),
            ("checkpoints/flux1-fill-dev-OneReward_fp8.safetensors", 11902532704),
            ("checkpoints/flux-hyp8-Q5_K_M.gguf", 8421981408),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("vae/ae.safetensors", 335304388),
            ("inpaint/sam_vit_h_4b8939.pth", 2564550879),
            ("style_models/flux1-redux-dev.safetensors", 129063232),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("rembg/General.safetensors", 884878856),
            ("loras/comfyui_subject_lora16.safetensors", 153268392),
            ("llms/Helsinki-NLP/opus-mt-zh-en/config.json", 1394),
            ("llms/Helsinki-NLP/opus-mt-zh-en/generation_config.json", 293),
            ("llms/Helsinki-NLP/opus-mt-zh-en/metadata.json", 1477),
            ("llms/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin", 312087009),
            ("llms/Helsinki-NLP/opus-mt-zh-en/source.spm", 804677),
            ("llms/Helsinki-NLP/opus-mt-zh-en/target.spm", 806530),
            ("llms/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json", 44),
            ("llms/Helsinki-NLP/opus-mt-zh-en/vocab.json", 1617902),
        ],
        "download_links": [
        "【选配】https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/flux1-fill-dev-OneReward_fp8.safetensors"
        ]
    },
        "eraser-a_package": {
        "id": 12,
        "name": "[12]一键消除",
        "note": "一键消除-默认模型[Flux1-fill-dev-OneReward]|显存需求：★★ 速度：★★☆",
        "files": [
            ("checkpoints/flux1-fill-dev-OneReward_fp8.safetensors", 11902532704),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("vae/ae.safetensors", 335304388),
            ("loras/removal_timestep_alpha-2-1740.safetensors",89746016),
            ("llms/Helsinki-NLP/opus-mt-zh-en/config.json", 1394),
            ("llms/Helsinki-NLP/opus-mt-zh-en/generation_config.json", 293),
            ("llms/Helsinki-NLP/opus-mt-zh-en/metadata.json", 1477),
            ("llms/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin", 312087009),
            ("llms/Helsinki-NLP/opus-mt-zh-en/source.spm", 804677),
            ("llms/Helsinki-NLP/opus-mt-zh-en/target.spm", 806530),
            ("llms/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json", 44),
            ("llms/Helsinki-NLP/opus-mt-zh-en/vocab.json", 1617902),
        ],
        "download_links": [
        "【选配】一键消除基于FluxAIO组件扩展，请检查所需模型包。"
        ]
    },
        "Illustrious_package": {
        "id": 13,
        "name": "[13]光辉模型包",
        "note": "支持NoobAI/光辉文生图-默认模型[miaomiaoV1.5b]|显存需求：★★ 速度：★★★☆",
        "files": [
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202)
        ],
        "download_links": []
    },
        "Illustrious_aio_package": {
        "id": 14,
        "name": "[14]光辉AIO扩展包",
        "note": "NoobAI/光辉全功能-默认模型[miaomiaoV1.5b]|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202),
            ("ipadapter/noob_ip_adapter.bin", 1396798350),
            ("upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("controlnet/noob_sdxl_controlnet_inpainting.safetensors", 5004167832),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408)
        ],
        "download_links": []
    },
        "StyleTransfer_package": {
        "id": 15,
        "name": "[15]风格转绘扩展包",
        "note": "多种图像风格转绘|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/LEOSAM_HelloWorldXL_70.safetensors", 6938040682),
            ("checkpoints/miaomiaoHarem_v15b.safetensors", 6938043202),
            ("checkpoints/SDXL_Yamers_Cartoon_Arcadia.safetensors", 6938040714),
            ("loras/SDXL_claymate.safetensors", 912561180),
            ("loras/SDXL_crayon.safetensors", 340776492),
            ("loras/SDXL_cute.safetensors", 681244276),
            ("loras/SDXL_ghibli.safetensors", 681268820),
            ("loras/SDXL_inkpainting.safetensors", 228466036),
            ("loras/SDXL_oilpainting.safetensors", 202694420),
            ("loras/SDXL_papercut.safetensors", 456489140),
            ("loras/SDXL_watercolor.safetensors", 228458788),
            ("loras/Illustrious_pixelart.safetensors", 228504612),
            ("loras/noob_pvc.safetensors", 607394012),
            ("loras/SDXL_lineart.safetensors", 170540028),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/lllyasviel/Annotators/sk_model.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/sk_model2.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/ControlNetHED.pth", 29444406),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("ipadapter/ip-adapter-faceid-plusv2_sdxl.bin", 1487555181),
            ("ipadapter/noob_ip_adapter.bin", 1396798350),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("insightface/models/buffalo_l/1k3d68.onnx", 143607619),
            ("insightface/models/buffalo_l/2d106det.onnx", 5030888),
            ("insightface/models/buffalo_l/det_10g.onnx", 16923827),
            ("insightface/models/buffalo_l/genderage.onnx", 1322532),
            ("insightface/models/buffalo_l/w600k_r50.onnx", 174383860)
        ],
        "download_links": []
    },
        "okdepthstatue_package": {
        "id": 16,
        "name": "[16]深度图、雕像扩展包",
        "note": "深度图、白瓷雕像风格扩展|显存需求：★★ 速度：★★★★★",
        "files": [
            ("checkpoints/juggernautXL_juggXIByRundiffusion.safetensors", 7105350536),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("controlnet/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth", 1341395338),
            ("controlnet/lllyasviel/Annotators/sk_model.pth", 17173511),
            ("controlnet/lllyasviel/Annotators/sk_model2.pth", 17173511),
            ("clip_vision/clip_vision_vit_h.safetensors", 1972298538),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427)
        ],
        "download_links": []
    },
        "Illustrious2_aio_package": {
        "id": 17,
        "name": "[17]光辉2.0_AIO扩展包",
        "note": "NoobAI/光辉2.0全功能-默认模型oneObsV13|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/oneObsession_13.safetensors", 6938040682),
            ("ipadapter/noob_ip_adapter.bin", 1396798350),
            ("upscale_models/RealESRGAN_x4plus_anime_6B.pth", 17938799),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("controlnet/noob_sdxl_controlnet_inpainting.safetensors", 5004167832),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408)
        ],
        "download_links": []
    },
        "nunchaku_int4_aio_package": {
        "id": 18,
        "name": "[18]双截棍int4量化Flux扩展包",
        "note": "适配非50系-默认模型[svdq-int4]|显存需求：★★★ 速度：★★★",
        "files": [
            ("checkpoints/svdq-int4_r32-flux.1-dev.safetensors", 6768309832),
            ("checkpoints/svdq-int4_r32-flux.1-fill-dev.safetensors", 6770275936),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": []
    },
        "nunchaku_fp4_aio_package": {
        "id": 19,
        "name": "[19]双截棍fp4量化Flux扩展包",
        "note": "仅适配50系-默认模型[svdq-fp4]|显存需求：★★☆ 速度：★★★",
        "files": [
            ("checkpoints/svdq-fp4_r32-flux.1-dev.safetensors", 7038706888),
            ("checkpoints/svdq-fp4_r32-flux.1-fill-dev.safetensors", 7040672992),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/EVA02_CLIP_L_336_psz14_s6B.pt", 856461210),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("clip_vision/sigclip_vision_patch14_384.safetensors", 856505640),
            ("controlnet/flux.1-dev_controlnet_union_pro_2.0.safetensors", 4281779224),
            ("controlnet/flux.1-dev_controlnet_upscaler.safetensors", 3583232168),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("insightface/models/antelopev2/1k3d68.onnx", 143607619),
            ("insightface/models/antelopev2/2d106det.onnx", 5030888),
            ("insightface/models/antelopev2/genderage.onnx", 1322532),
            ("insightface/models/antelopev2/glintr100.onnx", 260665334),
            ("insightface/models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
            ("loras/flux1-depth-dev-lora.safetensors", 1244440512),
            ("pulid/pulid_flux_v0.9.1.safetensors", 1142099520),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("vae/ae.safetensors", 335304388),
            ("style_models/flux1-redux-dev.safetensors", 129063232)
        ],
        "download_links": []
    },
    "kontext_package": {
        "id": 20,
        "name": "[20]Flux_Kontext扩展包",
        "note": "Flux_Kontext指令修图功能扩展包|显存需求：★★★☆ 速度：★★",
        "files": [
            ("checkpoints/flux1-dev-kontext_fp8_scaled.safetensors", 11904640136),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("loras/flux1-turbo.safetensors", 694082424),
            ("vae/ae.safetensors", 335304388)
        ],
        "download_links": [
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/flux1-dev-kontext_fp8_scaled.safetensors"
        ]
    },
    "wan_t2i_package": {
        "id": 21,
        "name": "[21]Wan2.2_T2I扩展包",
        "note": "万相2.2文生图扩展包|显存需求：★★★ 速度：★★",
        "files": [
            ("checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf", 11629612832),
            ("clip/umt5-xxl-encoder-Q8_0.gguf", 6043068256),
            ("vae/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", 316822496),
            ("loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", 316822496),
            ("loras/WAN2.1_SmartphoneSnapshotPhotoReality_v1_by-AI_Characters.safetensors", 306848672),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
        ],
        "download_links": [
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip/umt5-xxl-encoder-Q8_0.gguf"
        ]
    },
    "LSnet_package": {
        "id": 22,
        "name": "[22]LSnet画师串反推器",
        "note": "用于反推二次元风格对应画师名|显存需求：★ 速度：★★★★★",
        "files": [
            ("lsnet/kaloscope/https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/best_checkpoint.pth", 2015978609),
            ("lsnet/kaloscope/https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/class_mapping.csv", 574531)
        ],
        "download_links": [
        ]
    },
    "wan_i2v_package": {
        "id": 23,
        "name": "[23]Wan2.2图生视频扩展包",
        "note": "通义万相2.2图生图扩展包|显存需求：★★★★ 速度：★",
        "files": [
            ("checkpoints/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", 9651728896),
            ("checkpoints/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", 9651728896),
            ("clip/umt5-xxl-encoder-Q8_0.gguf", 6043068256),
            ("vae/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("loras/https://www.modelscope.cn/models/lightx2v/Wan2.2-Distill-Loras/resolve/master/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors", 634645944),
            ("loras/https://www.modelscope.cn/models/lightx2v/Wan2.2-Distill-Loras/resolve/master/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors", 739472104),
            ("controlnet/rife/https://www.modelscope.cn/models/windecay/rife/resolve/master/flownet.pkl", 24636301)
        ],
        "download_links": [
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip/umt5-xxl-encoder-Q8_0.gguf"
        ]
    },
    "wan_t2v_package": {
        "id": 24,
        "name": "[24]Wan2.2文生视频扩展包",
        "note": "通义万相2.2文生视频扩展包|显存需求：★★★★ 速度：★",
        "files": [
            ("checkpoints/Wan2.2_T2V_High_Noise_14B_VACE-Q4_K_M.gguf", 11629612832),
            ("checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf", 11629612832),
            ("clip/umt5-xxl-encoder-Q8_0.gguf", 6043068256),
            ("vae/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", 630697104),
            ("controlnet/rife/https://www.modelscope.cn/models/windecay/rife/resolve/master/flownet.pkl", 24636301)
        ],
        "download_links": [
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2_T2V_High_Noise_14B_VACE-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip/umt5-xxl-encoder-Q8_0.gguf"
        ]
    },
    "onekey_kontext_package": {
        "id": 25,
        "name": "[25]OneKeyKontext一键精修预置包",
        "note": "基于Flux_Kontext的一键精修|显存需求：★★★★ 速度：★★☆",
        "files": [
            ("checkpoints/flux1-dev-kontext_fp8_scaled.safetensors", 11904640136),
            ("clip/clip_l.safetensors", 246144152),
            ("clip/t5xxl_fp8_e4m3fn.safetensors", 4893934904),
            ("loras/flux1-turbo.safetensors", 694082424),
            ("vae/ae.safetensors", 335304388),
            ("upscale_models/4x_NMKD-Siax_200k.safetensors", 66864028),
            ("loras/Kontext_general_V1.safetensors", 306593008),
            ("loras/Kontext_all.safetensors", 306593008),
            ("loras/Kontext_appliances_V1.safetensors", 343806368),
            ("loras/Kontext_makeup_V1.safetensors", 171970336),
            ("loras/Kontext_metal_V1.safetensors", 343806368),
            ("loras/Kontext_clothing_V1.safetensors", 306593008),
            ("loras/Kontext_jewelry_V1.safetensors", 306593008),
            ("loras/Kontext_digital3C.safetensors", 306593008),
            ("loras/Kontext_composite.safetensors", 343806400),
            ("loras/Kontext_pattern.safetensors", 343806384),
            ("loras/Kontext_scene_alpha.safetensors", 343806392),
            ("loras/Kontext_face_V1.safetensors", 306593008),
            ("loras/Kontext_angle_beta.safetensors", 343806392),
            ("loras/Kontext_3view.safetensors", 306593008),
            ("loras/Kontext_remove_V1.safetensors", 306593008),
            ("loras/Kontext_body_restore.safetensors", 343806408),
            ("loras/Kontext_takeclothes_V2.safetensors", 343806408),
            ("loras/Kontext_put_it_here_V4.2.safetensors", 358706112),
            ("loras/Kontext_deblur.safetensors", 306793968),
            ("loras/Kontext_depth_referencel.safetensors", 343806456)
        ],
        "download_links": []
    },
    "qwen_aio_package": {
        "id":26,
        "name": "[26]Qwen_Image2512全功能预置包",
        "note": "Qwen_Image2512全功能预置包|显存需求：★★★★★ 速度:★★",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/master/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", 20430679144),
            ("controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors", 3536027816),
            ("controlnet/lllyasviel/Annotators/ZoeD_M12_N.pt", 1443406099),
            ("controlnet/parsing_bisenet.pth", 53289463),
            ("upscale_models/4x-UltraSharp.pth", 66961958),
            ("clip/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
            ("vae/qwen_image_vae.safetensors", 253806246),
            ("loras/https://www.modelscope.cn/models/lightx2v/Qwen-Image-2512-Lightning/resolve/master/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", 849608296),
            ("upscale_models/4xNomosUniDAT_bokeh_jpg.safetensors", 154152604),
            ("controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors", 4234599432)
        ],
        "download_links": []
    },
    "qwen_image_edit_plus_package": {
        "id":27,
        "name": "[27]QwenEdit+2511图像编辑预置包(含多视角Lora)",
        "note": "Qwen_Image_Edit+2511指令编辑图像|显存需求：★★★★★ 速度:★☆",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", 20533762817),
            ("loras/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 849608296),
            ("loras/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", 849608296),
            ("loras/https://www.modelscope.cn/models/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/master/qwen-image-edit-2511-multiple-angles-lora.safetensors",295140688),
            ("clip/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
            ("vae/qwen_image_vae.safetensors", 253806246),
            ("controlnet/hr16/DWPose-TorchScript-BatchSize5/https://www.modelscope.cn/models/svjack/DWPose-TorchScript-BatchSize5/resolve/master/dw-ll_ucoco_384_bs5.torchscript.pt", 135059124),
            ("controlnet/yzd-v/DWPose/https://www.modelscope.cn/models/zhangjin/DWPose/resolve/master/yolox_l.onnx", 216746733),
        ],
        "download_links": []
    },
        "nun_int4_qwen_image_edit_plus_package": {
        "id":28,
        "name": "[28]双截棍int4-QwenEdit+图像编辑",
        "note": "Qwen_Image_EditPlus指令编辑图像|显存需求：★★★★ 速度:★★★",
        "files": [
            ("checkpoints/https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/resolve/master/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors", 12654443144),
            ("clip/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
            ("vae/qwen_image_vae.safetensors", 253806246)
        ],
        "download_links": []
    },
        "nun_fp4_qwen_image_edit_plus_package": {
        "id":29,
        "name": "[29]双截棍fp4-QwenEdit+图像编辑",
        "note": "Qwen_Image_EditPlus指令编辑图像|显存需求：★★★★ 速度:★★★",
        "files": [
            ("checkpoints/https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/resolve/master/svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors", 13081386856),
            ("clip/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
            ("vae/qwen_image_vae.safetensors", 253806246)
        ],
        "download_links": []
    },
        "MiniCPM_V45_package": {
        "id": 30,
        "name": "[30]MiniCPMv45反推扩展包",
        "note": "本地多模态大语言模型[反推、翻译、扩写]|显存需求：★★ 速度：★★",
        "files": [
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/added_tokens.json", 2862),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/config.json", 1995),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/configuration.json", 51),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/configuration_minicpm.py", 3367),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/generation_config.json", 268),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/image_processing_minicpmv.py", 20757),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/merges.txt", 1671853),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model-00001-of-00002.safetensors", 4827364414),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model-00002-of-00002.safetensors", 1699920944),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model.safetensors.index.json", 267079),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/modeling_minicpmv.py", 17679),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/modeling_navit_siglip.py", 41835),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/preprocessor_config.json", 714),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/processing_minicpmv.py", 11026),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/resampler.py", 11374),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/special_tokens_map.json", 12103),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenization_minicpmv_fast.py", 1647),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenizer.json", 11437868),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenizer_config.json", 25786),
            ("llms/MiniCPM-V-4_5-int4/https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/vocab.json", 2776833),
        ],
        "download_links": []
    },
        "sdxl_package": {
        "id":31,
        "name": "[31]怀旧fooocus-SDXL支持包",
        "note": "fooocus后端SDXL模块支持包|显存需求：★★ 速度:★★★☆",
        "files": [
            ("controlnet/ip-adapter-plus-face_sdxl_vit-h.bin", 1013454761),
            ("controlnet/ip-adapter-plus_sdxl_vit-h.bin", 1013454427),
            ("controlnet/xinsir_cn_union_sdxl_1.0_promax.safetensors", 2513342408),
            ("loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", 371842896), 
            ("loras/sdxl_lightning_4step_lora.safetensors", 393854592),
            ("upscale_models/fooocus_upscaler_s409985e5.bin", 33636613),
            ("loras/Hyper-SDXL-8steps-lora.safetensors", 787359648),
            ("embeddings/unaestheticXLhk1.safetensors", 33296),
            ("embeddings/unaestheticXLv31.safetensors", 33296),
            ("inpaint/inpaint_v26.fooocus.patch", 1323362033),
            ("inpaint/inpaint_v25.fooocus.patch", 2580722369),
            ("llms/nllb-200-distilled-600M/pytorch_model.bin", 2460457927),
            ("llms/nllb-200-distilled-600M/sentencepiece.bpe.model", 4852054),
            ("llms/nllb-200-distilled-600M/tokenizer.json", 17331176),
            ("prompt_expansion/fooocus_expansion/config.json", 937),
            ("prompt_expansion/fooocus_expansion/merges.txt", 456356),
            ("prompt_expansion/fooocus_expansion/positive.txt", 5655),
            ("prompt_expansion/fooocus_expansion/pytorch_model.bin", 351283802),
            ("prompt_expansion/fooocus_expansion/special_tokens_map.json", 99),
            ("prompt_expansion/fooocus_expansion/tokenizer.json", 2107625),
            ("prompt_expansion/fooocus_expansion/tokenizer_config.json", 255),
            ("prompt_expansion/fooocus_expansion/vocab.json", 798156),
            ("safety_checker/stable-diffusion-safety-checker.bin", 1216067303),
        ],
        "download_links": []
    },
    "wan_ttp_package": {
        "id": 32,
        "name": "[32]Wan2.2_TTP超清放大扩展包",
        "note": "万相2.2TTP超清放大扩展包|显存需求：★★★ 速度：★★",
        "files": [
            ("checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf", 11629612832),
            ("clip/umt5-xxl-encoder-Q8_0.gguf", 6043068256),
            ("vae/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("vae/https://www.modelscope.cn/models/spacepxl/Wan2.1-VAE-upscale2x/resolve/master/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",507684560),
            ("loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", 630697104),
            ("SEEDVR2/https://www.modelscope.cn/models/numz/SeedVR2_comfyUI/resolve/master/ema_vae_fp16.safetensors", 501324814),
            ("SEEDVR2/https://www.modelscope.cn/models/numz/SeedVR2_comfyUI/resolve/master/seedvr2_ema_3b_fp16.safetensors", 6783018808)
        ],
        "download_links": [
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/checkpoints/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf",
            "https://www.modelscope.cn/models/metercai/SimpleSDXL2/resolve/master/SimpleModels/clip/umt5-xxl-encoder-Q8_0.gguf"
        ]
    },
    "newbie_image_package": {
        "id": 33,
        "name": "[33]NewbieImage扩展包",
        "note": "NewbieImage二次元大模型|显存需求：★★★ 速度：★★",
        "files": [
            ("unet/https://www.modelscope.cn/models/windecay/Models/resolve/master/newbieImage_exp01Base.safetensors", 6973329400),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/config.json", 2152),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/config_sentence_transformers.json", 281),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/configuration.json", 76),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/custom_st.py", 11988),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/model.safetensors", 1730688642),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/modules.json", 273),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/preprocessor_config.json", 584),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/special_tokens_map.json", 964),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/tokenizer.json", 17082997),
            ("jina_clip/https://www.modelscope.cn/models/jinaai/jina-clip-v2/resolve/master/tokenizer_config.json", 1148),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/added_tokens.json", 35),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/chat_template.json", 1615),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/config.json", 855),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/configuration.json", 76),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/generation_config.json", 215),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/model-00001-of-00002.safetensors", 4961251752),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/model-00002-of-00002.safetensors", 3639026128),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/model.safetensors.index.json", 90558),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/preprocessor_config.json", 570),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/special_tokens_map.json", 662),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/tokenizer.json", 33384568),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/tokenizer.model", 4689074),
            ("gemma3/https://www.modelscope.cn/models/google/gemma-3-4b-it/resolve/master/tokenizer_config.json", 1156999),
            ("vae/ae.safetensors", 335304388),
            ("upscale_models/https://modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/upscale_models/4x-AnimeSharp.pth",67010245)
        ],
        "download_links": []
    },
    "wan_scail_package": {
        "id": 34,
        "name": "[34]Wan_SCAIL扩展包",
        "note": "万相_SCAIL动作迁移扩展包|显存需求：★★★ 速度：★",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/Kijai/WanVideo_comfy_fp8_scaled/resolve/master/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors", 16401525232),
            ("clip/umt5-xxl-encoder-Q8_0.gguf", 6043068256),
            ("vae/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 738005744),
            ("controlnet/rife/https://www.modelscope.cn/models/windecay/rife/resolve/master/flownet.pkl", 24636301),
            ("detection/https://www.modelscope.cn/models/Kijai/vitpose_comfy/resolve/master/onnx/vitpose_h_wholebody_data.bin", 2548958740),
            ("detection/https://www.modelscope.cn/models/Kijai/vitpose_comfy/resolve/master/onnx/vitpose_h_wholebody_model.onnx", 420252),
            ("detection/https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B/resolve/master/process_checkpoint/det/yolov10m.onnx", 61659339),
            ("nlf/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/nlf/nlf_l_multi_0.3.2.torchscript", 493117974)
        ],
        "download_links": []
    },
    "zit_ttp_package": {
        "id": 35,
        "name": "[35]Z-Image_Turbo_TTP超清放大扩展包",
        "note": "Z-Image_Turbo_TTP超清放大|显存需求：★★★ 速度：★★",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/VerStella/z_image_turbo_comfyui/resolve/master/split_files/diffusion_models/z_image_turbo_bf16.safetensors", 12309866400),
            ("text_encoders/https://www.modelscope.cn/models/VerStella/z_image_turbo_comfyui/resolve/master/split_files/text_encoders/qwen_3_4b.safetensors", 8044982048),
            ("vae/ae.safetensors", 335304388),
            ("vae/https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/SimpleModels/vae/UltraFlux-vae_v1.safetensors", 335306212),
            ("SEEDVR2/https://www.modelscope.cn/models/numz/SeedVR2_comfyUI/resolve/master/ema_vae_fp16.safetensors", 501324814),
            ("SEEDVR2/https://www.modelscope.cn/models/numz/SeedVR2_comfyUI/resolve/master/seedvr2_ema_3b_fp16.safetensors", 6783018808)
        ],
        "download_links": []
    }, 
    "flux2-klein-9b-fp8": {
        "id": 36,
        "name": "[36]Flux2-Klein-9B-FP8",
        "note": "Flux2-Klein-9B-FP8|显存需求：★★★ 速度：★★★",
        "files": [
            ("diffusion_models/https://www.modelscope.cn/models/Comfy-Org/flux2-klein-9B/resolve/master/split_files/diffusion_models/flux-2-klein-9b-fp8.safetensors", 9433061528),
            ("text_encoders/https://www.modelscope.cn/models/Comfy-Org/flux2-klein-9B/resolve/master/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", 8664848742),
            ("vae/https://www.modelscope.cn/models/Comfy-Org/flux2-klein-4B/resolve/master/split_files/vae/flux2-vae.safetensors", 336213556),
            ("controlnet/hr16/DWPose-TorchScript-BatchSize5/https://www.modelscope.cn/models/svjack/DWPose-TorchScript-BatchSize5/resolve/master/dw-ll_ucoco_384_bs5.torchscript.pt", 135059124),
            ("controlnet/yzd-v/DWPose/https://www.modelscope.cn/models/zhangjin/DWPose/resolve/master/yolox_l.onnx", 216746733),
        ],
        "download_links": []
    }
}

MANUAL_DOWNLOAD_MAP = {
    "checkpoints": [
        "animaPencilXL_v500.jpg",
        "flux1-dev.jpg",
        "flux1-dev-fp8.jpg",
        "flux1-fill-dev_fp8.jpg",
        "flux-hyp8-Q5_K_M.jpg",
        "hunyuan_dit_1.2.jpg",
        "juggernautXL_juggXIByRundiffusion.jpg",
        "LEOSAM_HelloWorldXL_70.jpg",
        "miaomiaoHarem_v15b.jpg",
        "playground-v2.5-1024px.jpg",
        "ponyDiffusionV6XL.jpg",
        "realisticStockPhoto_v20.jpg",
        "realisticVisionV60B1_v51VAE.jpg",
        "sd3.5_large.jpg",
        "sd3.5_medium_incl_clips_t5xxlfp8scaled.jpg",
        "sd3_medium_incl_clips_t5xxlfp8.jpg",
        "SDXL_Yamers_Cartoon_Arcadia.jpg"
    ],
    "loras": [
        "comfyui_portrait_lora64.jpg",
        "comfyui_subject_lora16.jpg",
        "fill_remove.jpg",
        "FilmVelvia3.jpg",
        "flux_graffiti_v1.jpg",
        "flux1-depth-dev-lora.jpg",
        "flux1-turbo.jpg",
        "Hyper-SDXL-8steps-lora.jpg",
        "ip-adapter-faceid-plusv2_sd15_lora.jpg",
        "ip-adapter-faceid-plusv2_sdxl_lora.jpg",
        "sd_xl_offset_example-lora_1.0.jpg",
        "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.jpg",
        "sdxl_hyper_sd_4step_lora.jpg",
        "sdxl_lightning_4step_lora.jpg",
        "StickersRedmond.jpg",
        "Illustrious_pixelart.jpg",
        "noob_pvc.jpg",
        "SDXL_claymate.jpg", 
        "SDXL_crayon.jpg",
        "SDXL_cute.jpg",
        "SDXL_ghibli.jpg",
        "SDXL_inkpainting.jpg",
        "SDXL_lineart.jpg",
        "SDXL_oilpainting.jpg",
        "SDXL_papercut.jpg",
        "SDXL_watercolor.jpg"
    ]
}

MANUAL_DOWNLOAD_LIST = [
    f"https://hf-mirror.com/windecay/SimpleSDXL2/resolve/main/SimpleModels/{category}/{filename}"
    for category, files in MANUAL_DOWNLOAD_MAP.items() 
    for filename in files
]

OBSOLETE_MODELS = [
    "flux1-dev-bnb-nf4.safetensors",
    "flux1-dev-bnb-nf4-v2.safetensors",
    "flux1-schnell-bnb-nf4.safetensors",
    "sdxl_hyper_sd_4step_lora.safetensors",
    "xinsir_cn_openpose_sdxl_1.0.safetensors",
    "fooocus_xl_cpds_128.safetensors",
    "control-lora-canny-rank128.safetensors",
    "flux1-canny-dev-lora.safetensors",
    "flux.1-dev_controlnet_union_pro.safetensors",
    "illustriousXL_controlnet_tile_v2.5.safetensors",
    "kolors_controlnet_canny.safetensors",
    "kolors_controlnet_depth.safetensors",
    "noob_sdxl_controlnet_canny.fp16.safetensors",
    "noob_sdxl_controlnet_depth.fp16.safetensors",
    "noob_sdxl_controlnet_pose.fp16.safetensors",
    "NoobAI-XL-v1.1.safetensors",
    "clip-vit-h-14-laion2B-s32B-b79K.safetensors",
    "fill_remove.safetensors",
    "Qwen-Image-Lightning-8steps-V1.0.safetensors",
    "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors",
    "hunyuan_dit_1.2.safetensors",
    "playground-v2.5-1024px.safetensors",
    "ponyDiffusionV6XL.safetensors",
    "realisticStockPhoto_v20.safetensors",
    "sd3_medium_incl_clips_t5xxlfp8.safetensors",
    "flux1-fill-dev-hyp8-Q4_K_S.gguf",
    "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
    "sd3x_fp16.vae.safetensors",
    "sd3.5_large.safetensors",
    "clip_g.safetensors",
    "StickersRedmond.safetensors",
    "flux1-fill-dev_fp8.safetensors",
    "Qwen_Image_Edit-Q4_K_M.gguf",
    "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
    "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
    "FramePackI2V_HY_fp8_e4m3fn.safetensors"
    "llava_llama3_fp8_scaled.safetensors",
    "hunyuan_video_vae_bf16.safetensors",
    "kolors_unet_fp16.safetensors",
    "kolors_clip_ipa_plus_vit_large_patch14_336.bin",
    "kolors_controlnet_pose.safetensors",
    "kolors_ipa_faceid_plus.bin",
    "kolors_ip_adapter_plus_general.bin",
    "pytorch_model-00001-of-00007.bin",
    "pytorch_model-00002-of-00007.bin",
    "pytorch_model-00003-of-00007.bin",
    "pytorch_model-00004-of-00007.bin",
    "pytorch_model-00005-of-00007.bin",
    "pytorch_model-00006-of-00007.bin",
    "pytorch_model-00007-of-00007.bin",
    "FilmVelvia3.safetensors",
    "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors",
    "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
    "Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors",
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
    "qwen-image-Q4_K_M.gguf"
]

MODELSCOPE_FILE_CACHE = {}

def get_modelscope_file_sha256(url, verbose=True):
    """
    尝试从ModelScope API获取文件的SHA256
    URL格式: https://www.modelscope.cn/models/{namespace}/{repo_name}/resolve/{revision}/{file_path}
    API格式: https://modelscope.cn/api/v1/models/{namespace}/{repo_name}/repo/files?Revision={revision}&Recursive=true
    """
    pattern = r'https?://(?:www\.)?modelscope\.cn/models/([^/]+)/([^/]+)/resolve/([^/]+)/(.*)'
    match = re.match(pattern, url)

    if not match:
        return None

    namespace, repo_name, revision, file_path = match.groups()
    cache_key = (namespace, repo_name, revision)

    try:
        from urllib.parse import unquote
        file_path = unquote(file_path)
    except:
        pass

    if cache_key not in MODELSCOPE_FILE_CACHE:
        api_url = f"https://modelscope.cn/api/v1/models/{namespace}/{repo_name}/repo/files?Revision={revision}&Recursive=true"
        try:
            if verbose:
                print(f"{Fore.CYAN}正在获取官方校验数据: {namespace}/{repo_name} ({revision})...{Style.RESET_ALL}")
            # 设置较短超时，避免卡住
            response = requests.get(api_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                file_map = {}
                if 'Data' in data and 'Files' in data['Data']:
                    for f in data['Data']['Files']:
                        if f['Type'] == 'blob':
                            file_map[f['Path']] = f['Sha256']
                MODELSCOPE_FILE_CACHE[cache_key] = file_map
                if verbose:
                    print(f"{Fore.GREEN}√ 获取成功，已缓存 {len(file_map)} 个文件的特征值{Style.RESET_ALL}")
            else:
                if verbose:
                    print(f"{Fore.RED}无法获取官方数据 (HTTP {response.status_code}){Style.RESET_ALL}")
                MODELSCOPE_FILE_CACHE[cache_key] = None
        except Exception as e:
            if verbose:
                print(f"{Fore.RED}获取官方数据失败: {e}{Style.RESET_ALL}")
            MODELSCOPE_FILE_CACHE[cache_key] = None

    file_map = MODELSCOPE_FILE_CACHE.get(cache_key)
    if file_map:
        return file_map.get(file_path)
    return None

def calculate_sha256(file_path):
    """计算文件的SHA256哈希值"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"Error: {e}"

def verify_package_strict(package_id, packages):
    """严格校验包内文件的SHA256"""
    # Find package
    target_package = None
    for pkg_name, pkg_info in packages.items():
        if pkg_info["id"] == package_id:
            target_package = pkg_info
            break

    if not target_package:
        print(f"{Fore.RED}△未找到ID为 {package_id} 的模型包{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}正在严格校验模型包: {target_package['name']} (计算SHA256需要时间，请耐心等待)...{Style.RESET_ALL}")

    path_mapping = load_model_paths()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    files_and_sizes = target_package["files"]
    corrupted_files = []

    for expected_path, expected_size in files_and_sizes:
        expected_filename = os.path.basename(expected_path)
        path_parts = expected_path.split('/')
        path_type = path_parts[0] if len(path_parts) > 0 else ''
        sub_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''

        search_dirs = sorted(
            path_mapping.get(path_type, []),
            key=lambda x: (
                0 if "SimpleModels" in x else
                1 if any(part == "models" for part in x.split(os.sep)) else
                2,
                x
            )
        )
        if not search_dirs:
            simplemodels_default = os.path.join(root, "SimpleModels")
            search_dirs = [os.path.join(simplemodels_default, path_type)]

        found = False
        actual_path = None

        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        url_match = re.search(url_pattern, expected_path)
        
        target_url = None
        if url_match:
            target_url = url_match.group(0)
            local_dir = expected_path.split(target_url)[0].rstrip('/')
            file_name = os.path.basename(target_url)

            for base_dir in search_dirs:
                actual_full_path = os.path.join(base_dir, local_dir.replace(path_type, "", 1).lstrip('/'), file_name)
                actual_full_path = os.path.normpath(actual_full_path)

                if os.path.exists(actual_full_path):
                    actual_path = actual_full_path
                    found = True
                    break
        else:
            target_url = f"{DEFAULT_DOWNLOAD_PREFIX}SimpleModels/{expected_path}"

        if not found:
            for base_dir in search_dirs:
                full_path = os.path.join(base_dir, sub_path) if sub_path else os.path.join(base_dir, os.path.basename(expected_path))
                if os.path.exists(full_path):
                    actual_path = full_path
                    found = True
                    break

        if found and actual_path:
            print(f"正在计算: {os.path.basename(actual_path)} ...", end="", flush=True)
            sha256_val = calculate_sha256(actual_path)
            print(f"\r{Fore.GREEN}文件: {os.path.basename(actual_path)}{Style.RESET_ALL}")
            print(f"  路径: {actual_path}")
            print(f"  SHA256: {Fore.YELLOW}{sha256_val}{Style.RESET_ALL}")
            print(f"  大小: {os.path.getsize(actual_path)} bytes (预期: {expected_size})")

            # 尝试获取官方SHA256
            official_sha256 = None
            if target_url:
                official_sha256 = get_modelscope_file_sha256(target_url)

            if official_sha256:
                if sha256_val.lower() == official_sha256.lower():
                     print(f"  校验结果: {Fore.GREEN}√ 通过 (与官方一致){Style.RESET_ALL}")
                else:
                     print(f"  校验结果: {Fore.RED}× 失败 (官方: {official_sha256}){Style.RESET_ALL}")
                     corrupted_files.append((actual_path, target_url, expected_size))
            else:
                 print(f"  校验结果: {Fore.YELLOW}? 未能获取官方数据，请人工比对{Style.RESET_ALL}")

        else:
            print(f"{Fore.RED}×文件缺失: {expected_path}{Style.RESET_ALL}")

    if corrupted_files:
        print(f"\n{Fore.RED}发现 {len(corrupted_files)} 个文件的SHA256与官方不匹配：{Style.RESET_ALL}")
        for path, _, _ in corrupted_files:
            print(f"- {path}")
        
        print(f"\n{Fore.YELLOW}是否删除这些受损文件并重新下载？(y/n): {Style.RESET_ALL}", end="")
        choice = input().strip().lower()
        if choice == 'y':
            with open("downloadlist.txt", "w") as f1:
                for path, url, size in corrupted_files:
                    try:
                        os.remove(path)
                        print(f"已删除: {path}")
                        f1.write(f"{url},{size}\n")
                    except Exception as e:
                        print(f"删除失败 {path}: {e}")
            
            print("启动自动下载...")
            auto_download_missing_files_with_retry()

    print(f"\n{Fore.CYAN}校验完成。{Style.RESET_ALL}")

def main():
    print()
    print_colored("★★★★★★★★★★★★★★★★★★欢迎使用SimpleAI模型检测器★★★★★★★★★★★★★★★★★★", Fore.CYAN)
    time.sleep(0.1)
    print()
    check_python_embedded()
    time.sleep(0.1)
    check_script_file()
    time.sleep(0.1)
    total_virtual = get_total_virtual_memory()
    time.sleep(0.1)
    check_virtual_memory(total_virtual)
    time.sleep(0.1)
    print_instructions()
    time.sleep(0.1)
    validate_files(packages)
    print()
    print_colored("★★★★★★★★★★★★★★★★★★检测已结束执行自动下载模块★★★★★★★★★★★★★★★★★★", Fore.CYAN)

if __name__ == "__main__":
    main()
    print()
    while True:
        print(f">>>输入【{Fore.YELLOW}ALL{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】----------------启动全部文件下载<<<     备注：支持断点续传，顺序从小文件开始。")
        print(f">>>输入【{Fore.YELLOW}模型包编号{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】----------启动预置包补全<<<     备注：可输入多个编号，例如1,5,7")
        print(f">>>数字【{Fore.YELLOW}0{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】-清理日志/下载/图片缓存与坏文件<<<     备注：△谨慎执行。慎防误删私有模型")
        print(f">>>输入【{Fore.YELLOW}DEL{Style.RESET_ALL}】【{Fore.YELLOW}模型包编号{Style.RESET_ALL}】----------删除已有模型包文件<<<     备注：△谨慎执行。自动避开关联文件")
        print(f">>>输入【{Fore.YELLOW}*DEL{Style.RESET_ALL}】【{Fore.YELLOW}模型包编号{Style.RESET_ALL}】-------强制删除模型包文件<<<     备注：△谨慎执行。不检查关联性直接删除")
        print(f">>>输入【{Fore.YELLOW}R{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】-----------------------重新检测<<<     备注：再玩一遍，玩不腻")
        print(f">>>输入【{Fore.YELLOW}S{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】-----------------下载模型预览图<<<     备注：只下载checkpoints和lora预览图")
        print(f">>>输入【{Fore.YELLOW}SHA{Style.RESET_ALL}】【{Fore.YELLOW}模型包编号{Style.RESET_ALL}】-------------校验模型包SHA256<<<     备注：严格校验,支持多个编号")
        print(f">>>输入【{Fore.YELLOW}H{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】--------切换下载源到Huggingface<<<     备注：当前使用源：{current_source}")
        print(f">>>输入【{Fore.YELLOW}M{Style.RESET_ALL}】 【{Fore.YELLOW}回车{Style.RESET_ALL}】--------切换下载源到ModelScope<<<<     备注：当前使用源：{current_source}")
        print("请选择操作(不需要括号):", flush=True)  # 启动器场景：保留换行符以便立即显示
        user_input = input()

        stripped_input = user_input.strip()

        if stripped_input.lower() == "all":
            print("※启动自动下载模块,支持断点续传，关闭窗口可中断。")
            auto_download_missing_files_with_retry(max_threads=5)
        elif stripped_input.lower().startswith("sha"):
            input_content = stripped_input[3:].strip()
            # 支持逗号分隔的多个ID，如 "1,3,5" 或 "1，3，5"
            normalized_input = input_content.replace('，', ',')
            if ',' in normalized_input:
                pkg_ids = normalized_input.split(',')
                valid_ids = []
                for pid in pkg_ids:
                    pid = pid.strip()
                    if pid.isdigit():
                        valid_ids.append(int(pid))
                    elif pid: # 忽略空字符串
                         print(f"{Fore.RED}△无效的模型包编号：{pid}{Style.RESET_ALL}")

                for pid in valid_ids:
                    verify_package_strict(pid, packages)
            elif input_content.isdigit():
                verify_package_strict(int(input_content), packages)
            else:
                 print(f"{Fore.RED}△输入格式错误，请输入类似 sha 1 或 sha 1,3,5 来校验对应模型包。{Style.RESET_ALL}")
        elif ',' in stripped_input or '，' in stripped_input:
            selected_packages = {}
            normalized_input = stripped_input.replace('，', ',')
            package_ids = normalized_input.split(',')
            valid_input = True

            for pkg_id_str in package_ids:
                pkg_id_str = pkg_id_str.strip()
                if not pkg_id_str.isdigit():
                    print(f"{Fore.RED}△输入格式错误：'{pkg_id_str}' 不是有效的模型包编号{Style.RESET_ALL}")
                    valid_input = False
                    break

                package_id = int(pkg_id_str)
                found = False

                for package_name, package_info in packages.items():
                    if package_info["id"] == package_id:
                        selected_packages[package_name] = package_info
                        found = True
                        break

                if not found:
                    print(f"{Fore.RED}△模型包编号{package_id} 无效，请输入正确的模型包ID。{Style.RESET_ALL}")
                    valid_input = False
                    break

            if valid_input and selected_packages:
                print(f"{Fore.GREEN}√已选择 {len(selected_packages)} 个模型包，正在生成合并的下载列表...{Style.RESET_ALL}")
                get_download_links_for_package(selected_packages, "downloadlist.txt")
                auto_download_missing_files_with_retry(max_threads=5)
        elif stripped_input.isdigit():
            package_id = int(stripped_input)
            selected_package = None
            for package_name, package_info in packages.items():
                if package_info["id"] == package_id:
                    selected_package = package_info
                    break

            if selected_package:
                get_download_links_for_package({package_name: selected_package}, "downloadlist.txt")
                auto_download_missing_files_with_retry(max_threads=5)
            elif package_id == 0:
                delete_partial_files()
                delete_specific_image_files()
                delete_log_files()
            else:
                print(f"{Fore.RED}△模型包编号{package_id} 无效，请输入正确的模型包ID。{Style.RESET_ALL}")
        elif stripped_input.lower().startswith("*del"):
            try:
                path_mapping = load_model_paths()
                package_id_str = stripped_input[4:].strip()

                if not package_id_str.isdigit():
                    print(f"{Fore.RED}△输入格式错误，请输入类似 *del1 来强制删除对应模型包。{Style.RESET_ALL}")
                else:
                    package_id = int(package_id_str)
                    selected_package = None
                    selected_pkg_name = None

                    for pkg_name, pkg_info in packages.items():
                        if pkg_info["id"] == package_id:
                            selected_package = pkg_info
                            selected_pkg_name = pkg_name
                            break

                    if selected_package:
                        delete_package_force(selected_pkg_name, packages)
                    else:
                        print(f"{Fore.RED}△无效的模型包编号！{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}△强制删除过程中发生错误：{str(e)}{Style.RESET_ALL}")
        elif stripped_input.lower().startswith("del"):
            try:
                path_mapping = load_model_paths()
                package_id_str = stripped_input[3:].strip()

                if not package_id_str.isdigit():
                    print(f"{Fore.RED}△输入格式错误，请输入类似 del1 来删除对应模型包。{Style.RESET_ALL}")
                else:
                    package_id = int(package_id_str)
                    selected_package = None

                    for pkg_name, pkg_info in packages.items():
                        if pkg_info["id"] == package_id:
                            selected_package = pkg_info
                            break

                    if selected_package:
                        print(f"{Fore.YELLOW}△即将删除模型包：[{selected_package['name']}]{Style.RESET_ALL}")
                        delete_package(pkg_name, packages)
                    else:
                        print(f"{Fore.RED}△无效的模型包编号！{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}△删除过程中发生错误：{str(e)}{Style.RESET_ALL}")
        elif stripped_input.lower() == "r":
            print("重新检测文件...")
            validate_files(packages)
        elif stripped_input.lower() == "s":
            print("下载预览图...")
            trigger_manual_download()
        elif stripped_input.lower() == "h":
            CURRENT_DOWNLOAD_PREFIX = HF_DOWNLOAD_PREFIX
            current_source = "HuggingFace拥抱脸国外源"
            validate_files(packages)
            print(f"{Fore.GREEN}√下载源已切换到Huggingface：{CURRENT_DOWNLOAD_PREFIX}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}※提示：此切换只在本次运行有效，重启程序后将恢复默认设置。{Style.RESET_ALL}")
        elif stripped_input.lower() == "m":
            CURRENT_DOWNLOAD_PREFIX = DEFAULT_DOWNLOAD_PREFIX
            current_source = "ModelScope魔搭国内源"
            validate_files(packages)
            print(f"{Fore.GREEN}√下载源已切换到ModelScope：{CURRENT_DOWNLOAD_PREFIX}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}※提示：此切换只在本次运行有效，重启程序后将恢复默认设置。{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△无效的输入，请输入回车或有效的模型包编号（不需要括号）。{Style.RESET_ALL}")