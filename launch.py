import os
import ssl
import sys
import json
import importlib
import packaging.version
import platform
import re
import time
import shared
import fooocus_version
import comfy.comfy_version as comfy_version
import enhanced.version as version
import socket
import logging
import shutil
import torch
from pathlib import Path
from build_launcher import build_launcher, ready_checker, is_win32_standalone_build, python_embeded_path, download_if_updated
from modules.launch_util import is_installed, is_installed_version, run, python, run_pip, requirements_met, delete_folder_content, git_clone, index_url, extra_index_url, target_path_install, met_diff
from enhanced.logger import setup_logger, now_string, get_log_file
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
setup_logger(log_level='INFO')
logger = logging.getLogger(__name__)

logger.debug('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

OBSOLETE_CUSTOM_NODE_FOLDERS = (
    "ComfyUI-MultiGPU",
    "ComfyUI-SAM3",
    "ComfyUI-Newbie-Nodes",
    "x-flux-comfyui",
)

def cleanup_obsolete_custom_nodes():
    custom_nodes_root = os.path.join(root, "comfy", "custom_nodes")
    if not os.path.isdir(custom_nodes_root):
        return
    for folder_name in OBSOLETE_CUSTOM_NODE_FOLDERS:
        target_path = os.path.join(custom_nodes_root, folder_name)
        if not os.path.isdir(target_path):
            continue
        try:
            shutil.rmtree(target_path)
            logger.info(f"[Cleanup] Removed obsolete custom node folder: {target_path}")
        except Exception as e:
            logger.warning(f"[Cleanup] Failed to remove obsolete custom node folder: {target_path} ({e})")

# cleanup_obsolete_custom_nodes()

os.environ["SIMPAI_LOG_FILE"] = get_log_file()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["translators_default_region"] = "China"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

def _make_pip_env():
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_USER"] = "0"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env

def install_package_with_retry(pkg_name, pkg_version=None, description=None):
    """尝试安装包，先使用阿里源，如果失败则尝试使用清华源"""
    desc = description or f'Installing {pkg_name}'
    errdesc = f"Couldn't install {pkg_name}"

    try:
        if pkg_version:
            pkg_command = f'pip install -U --no-user {pkg_name}=={pkg_version} -i {index_url}'
        else:
            pkg_command = f'pip install -U --no-user {pkg_name} -i {index_url}'

        run(f'"{python}" -s -m {pkg_command}', desc, errdesc, custom_env=_make_pip_env(), live=True)
        return True
    except Exception as e:
        logger.warning(f"阿里源安装{pkg_name}失败: {str(e)}")
        logger.info("尝试使用清华源镜像...")

    try:
        if pkg_version:
            pkg_command = f'pip install -U --no-user {pkg_name}=={pkg_version} -i {extra_index_url}'
        else:
            pkg_command = f'pip install -U --no-user {pkg_name} -i {extra_index_url}'

        run(f'"{python}" -s -m {pkg_command}', desc, errdesc, custom_env=_make_pip_env(), live=True)
        return True
    except Exception as e:
        logger.error(f"使用清华源安装{pkg_name}失败: {str(e)}")
        return False

def ensure_nvidia_vfx_installed():
    target_ver = "0.1.0.1"
    module_name = "nvvfx"
    dist_candidates = ("nvidia-vfx", "nvidia_vfx")

    version_ok = False
    for dist in dist_candidates:
        try:
            if is_installed_version(dist, target_ver):
                version_ok = True
                break
        except Exception:
            continue

    is_ok = is_installed(module_name) and version_ok
    need_reinstall = not is_ok

    if not need_reinstall and platform.system() == "Linux":
        try:
            import nvvfx  # noqa: F401
        except Exception:
            print("nvidia-vfx detected but failed to import. Reinstalling for Linux...")
            need_reinstall = True

    if not need_reinstall:
        return

    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        logger.info("Skip nvidia-vfx install: only cp310 wheels are provided.")
        return

    vfx_url = None
    vfx_name = None
    if platform.system() == "Windows":
        vfx_url = "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nvidia_vfx/nvidia_vfx-0.1.0.1-cp310-cp310-win_amd64.whl"
        vfx_name = "nvidia_vfx-0.1.0.1-cp310-cp310-win_amd64.whl"
    elif platform.system() == "Linux":
        vfx_url = "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nvidia_vfx/nvidia_vfx-0.1.0.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        vfx_name = "nvidia_vfx-0.1.0.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

    if not vfx_url:
        logger.info("Skip nvidia-vfx install: unsupported platform.")
        return

    vfx_path = os.path.abspath(os.path.join(root, vfx_name))
    print("check nvidia-vfx...")
    has_update_vfx = download_if_updated(vfx_url, vfx_path)
    if has_update_vfx or need_reinstall:
        print(f"ready to install {vfx_path}")
        run(
            f'"{python}" -s -m pip install --no-user -U --force-reinstall --no-deps {vfx_path}',
            f"Install {vfx_path}",
            custom_env=_make_pip_env(),
            live=True,
        )

def ensure_descript_audiotools_installed():
    target_ver = "0.7.4"
    module_candidates = ("descript_audiotools", "audiotools")
    dist_candidates = ("descript-audiotools", "descript_audiotools")

    version_ok = False
    for dist in dist_candidates:
        try:
            if is_installed_version(dist, target_ver):
                version_ok = True
                break
        except Exception:
            continue

    module_ok = False
    for module_name in module_candidates:
        try:
            if is_installed(module_name):
                module_ok = True
                break
        except Exception:
            continue

    need_reinstall = not (module_ok and version_ok)
    if not need_reinstall:
        return

    whl_url = "https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/audiotools/descript_audiotools-0.7.4-py2.py3-none-any.whl"
    whl_name = "descript_audiotools-0.7.4-py2.py3-none-any.whl"
    whl_path = os.path.abspath(os.path.join(root, whl_name))
    print("check descript_audiotools...")
    has_update_whl = download_if_updated(whl_url, whl_path)
    if has_update_whl or need_reinstall:
        print(f"ready to install {whl_path}")
        run(
            f'"{python}" -s -m pip install --no-user -U --force-reinstall --no-deps {whl_path}',
            f"Install {whl_path}",
            custom_env=_make_pip_env(),
            live=True,
        )
def check_base_environment():
    print(f"{now_string()} Python {sys.version}")
    print(f"{now_string()} Fooocus version: {fooocus_version.version}")
    print(f"{now_string()} Comfyd version: {comfy_version.version}")
    print(f'{now_string()} {version.get_branch()} version: {version.get_simplesdxl_ver()}')
    print(f'{now_string()} ✦ | 兴趣使然的版本 | ✦ by冰華 ✦')

    base_pkg = "simpleai_base"
    ver_required = "0.3.40"
    REINSTALL_BASE = False #if '_dev' not in version.get_branch() else True
    base_branch = "dev"
    # if '--dev' in (sys.argv):
    #     base_branch = 'dev'
    base_url = f"https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/{base_branch}"
    #base_url = f"https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/libs/{base_branch}"
    base_file = {
        "Windows": f'simpleai_base-{ver_required}-cp310-cp310-win_amd64.whl',
        "Linux": f'simpleai_base-{ver_required}-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        "Darwin_arm64": f'simpleai_base-{ver_required}-cp310-cp310-macosx_11_0_arm64.whl',
        'Darwin_x86_64': f'simpleai_base-{ver_required}-cp310-cp310-macosx_10_12_x86_64.whl'
        }
    platform_os = platform.system()
    if platform.system() == 'Darwin':
        if platform.machine() == 'arm64':
            platform_os = 'Darwin_arm64'
        else:
            platform_os = 'Darwin_x86_64'
    else:
        platform_os = platform.system()

    base_path = os.path.abspath(os.path.join(root, f'enhanced/libs/{base_file[platform_os]}'))
    base_url = f'{base_url}/{base_file[platform_os]}'
    has_update_whl = download_if_updated(base_url, base_path)
    if has_update_whl or REINSTALL_BASE or not is_installed_version(base_pkg, ver_required):
        if os.path.exists(base_path):
            if not is_installed(base_pkg):
                run(f'"{python}" -s -m pip install --no-user {base_path}', f'Install {base_pkg} {ver_required}', custom_env=_make_pip_env())
            else:
                version_installed = importlib.metadata.version(base_pkg)
                if REINSTALL_BASE or packaging.version.parse(ver_required) != packaging.version.parse(version_installed):
                    logger.info(f"正在更新 {base_pkg}: {version_installed} -> {ver_required}")
                    run(f'"{python}" -s -m pip install --no-user -U {base_path}', f'Update {base_pkg} {ver_required}', custom_env=_make_pip_env())
        else:
            if not is_installed(base_pkg):
                logger.error(f"缺失必要的包 {base_pkg} 且下载失败，程序可能无法正常运行。请检查网络连接并重新启动。")
            else:
                logger.warning(f"无法下载更新包 {base_pkg}，将继续使用当前版本 {importlib.metadata.version(base_pkg)}。")

    if torch.__version__ == '2.9.1+cu130':
        logger.info(f'当前环境：PyTorch 2.9.1+CUDA 13.0. 50系以上显卡支持Nvfp4模型加速推理.')
        update_pkgs = [('comfyui_frontend_package', '1.42.8'), ('comfyui_workflow_templates', '0.9.44'), ('comfyui-embedded-docs', '0.4.3'), ('comfy-kitchen', '0.2.8'), ('comfy-aimdo', '0.2.12'), ('transformers', '4.57.6'), ('PyOpenGL', '3.1.10'), ('glfw', '2.10.0'), ('blake3', '1.0.8'), ('aiohttp', '3.13.3'),
        ('ninja', '1.11.1.4'), ('numpy', '1.26.4'), ('absl-py', '2.4.0'), ('flatbuffers', '25.12.19'), ('mediapipe', '0.10.32'), ('psd-tools', '1.14.1'), ('docstring-parser', '0.17.0'), ('fire', '0.7.1'), ('flatten-dict', '0.4.2'), ('grpcio', '1.80.0'), ('julius', '0.2.7'), ('markdown', '3.10.2'), ('markdown2', '2.5.5'), ('pystoi', '0.4.1'), ('randomname', '0.2.1'), ('tensorboard', '2.20.0'), ('tensorboard-data-server', '0.7.2'), ('torch-stoi', '0.2.3'), ('werkzeug', '3.1.7'), ('sounddevice', '0.5.5')]
        for (update_pkg_name, update_pkg_version) in update_pkgs:
            if not is_installed_version(update_pkg_name, update_pkg_version):
                success = install_package_with_retry(update_pkg_name, update_pkg_version)
                if not success:
                    logger.error(f"无法安装{update_pkg_name}，请检查网络状态")
        try:
            target_llama_ver = '0.3.30'
            is_llama_installed = is_installed_version('llama_cpp_python', target_llama_ver) or is_installed_version('llama_cpp_python', '0.3.30+cu130.basic')
            need_reinstall = not is_llama_installed

            if is_llama_installed and platform.system() == 'Linux':
                try:
                    import llama_cpp
                except:
                    print("llama_cpp_python detected but failed to import. Reinstalling for Linux...")
                    need_reinstall = True

            if need_reinstall:
                llama_url = None
                if platform.system() == 'Windows':
                    llama_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.30%2Bcu130.basic-cp310-cp310-win_amd64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.30+cu130.basic-cp310-cp310-win_amd64.whl'))
                elif platform.system() == 'Linux' and sys.version_info.major == 3 and sys.version_info.minor == 10:
                    llama_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.30%2Bcu130.basic-cp310-cp310-linux_x86_64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.30+cu130.basic-cp310-cp310-linux_x86_64.whl'))

                if llama_url:
                    print('check llama_cpp_python...')
                    has_update_llama = download_if_updated(llama_url, llama_path)
                    if has_update_llama or not is_installed_version('llama_cpp_python', target_llama_ver) or need_reinstall:
                        print(f'ready to install {llama_path}')
                        run(f'"{python}" -s -m pip install --no-user -U --force-reinstall --no-deps {llama_path}', f'Install {llama_path}', custom_env=_make_pip_env(), live=True)
        except Exception as e:
            print(f'Error installing llama_cpp_python: {str(e)}')
            print('Skipping llama_cpp_python installation and continuing...')

        try:
            ensure_nvidia_vfx_installed()
        except Exception as e:
            print(f'Error installing nvidia-vfx: {str(e)}')
            print('Skipping nvidia-vfx installation and continuing...')

        try:
            ensure_descript_audiotools_installed()
        except Exception as e:
            print(f'Error installing descript_audiotools: {str(e)}')
            print('Skipping descript_audiotools installation and continuing...')

    elif is_installed("sageattention"):

        update_pkgs = [('comfyui_frontend_package', '1.42.8'), ('comfyui_workflow_templates', '0.9.44'), ('comfyui-embedded-docs', '0.4.3'), ('transformers', '4.57.6'), ('bitsandbytes', '0.45.5'), ('accelerate', '1.10.1'), ('av', '14.2.0'), ('yarl', '1.18.0'), ('gguf', '0.14.0'),
                       ('sentencepiece', '0.2.0'), ('diffusers', '0.36.0'), ('huggingface_hub', '0.35.1'), ('peft', '0.17.1'), ('tokenizers', '0.22.1'), ('tiktoken', '0.11.0'), ('librosa', '0.11.0'), ('moviepy', '2.2.1'), ('piexif', '1.1.3'), ('deepdiff', '8.6.0'), ('pydantic', '2.12.2'),
                       ('GitPython', '3.1.45'), ('PyGithub', '2.8.1'), ('matrix-nio', '0.24.0'), ('toml', '0.10.2'), ('uv', '0.9.3'), ('clip-interrogator', '0.6.0'), ('simpleeval', '1.0.3'), ('compel', '2.3.0'), ('rotary-embedding-torch', '0.8.9'), ('hydra-core', '1.3.2'), ('uuid7', '0.1.0'), ('aiosqlite', '0.21.0'), ('configs','3.0.3'),
                       ('mmdet', '3.3.0'), ('mmengine', '0.10.7'), ('munkres', '1.1.4'), ('terminaltables', '3.1.10'), ('color-matcher', '0.6.0'), ('natsort', '8.4.0'), ('olefile', '0.47'), ('taichi', '1.7.4'), ('torchdiffeq', '0.2.5'), ('lark', '1.3.1'), ('comfy-kitchen', '0.2.8'), ('comfy-aimdo', '0.2.12'), ('PyOpenGL', '3.1.10'), ('glfw', '2.10.0'), ('blake3', '1.0.8'),
                       ('aiohttp', '3.13.3'), ('ninja', '1.11.1.4'), ('numpy', '1.26.4'), ('absl-py', '2.4.0'), ('flatbuffers', '25.12.19'), ('mediapipe', '0.10.32'), ('psd-tools', '1.14.1'), ('docstring-parser', '0.17.0'), ('fire', '0.7.1'), ('flatten-dict', '0.4.2'), ('grpcio', '1.80.0'), ('julius', '0.2.7'), ('markdown', '3.10.2'), ('markdown2', '2.5.5'), ('pystoi', '0.4.1'), ('randomname', '0.2.1'), ('tensorboard', '2.20.0'), ('tensorboard-data-server', '0.7.2'), ('torch-stoi', '0.2.3'), ('werkzeug', '3.1.7'), ('sounddevice', '0.5.5')]
        for (update_pkg_name, update_pkg_version) in update_pkgs:
            if not is_installed_version(update_pkg_name, update_pkg_version):
                success = install_package_with_retry(update_pkg_name, update_pkg_version)
                if not success:
                    logger.error(f"无法安装{update_pkg_name}，请检查网络状态")
        if not is_installed_version("facenet-pytorch", "2.6.0"):
            logger.info("Installing facenet-pytorch==2.6.0 with --no-deps")
            run_pip(f"install -U facenet-pytorch==2.6.0 --no-deps", "facenet-pytorch==2.6.0")
        try:
            is_torch29_cu130_nunchaku = is_installed_version('nunchaku', '1.2.1+cu13.0torch2.9')
            is_torch29_nunchaku = is_installed_version('nunchaku', '1.2.1+cu12.8torch2.9')
            is_torch27_nunchaku = is_installed_version('nunchaku', '1.0.2+torch2.7')

            need_nunchaku_install = not is_torch29_cu130_nunchaku and not is_torch29_nunchaku and not is_torch27_nunchaku

            if not need_nunchaku_install and platform.system() == 'Linux':
                try:
                    import nunchaku
                except:
                    print("nunchaku detected but failed to import. It might be a cross-platform conflict. Reinstalling for Linux...")
                    need_nunchaku_install = True

            if need_nunchaku_install:
                torch_version = torch.__version__
                print(f'Detected PyTorch version: {torch_version}')

                pkg_url = None
                target_ver = None
                if platform.system() == 'Windows':
                    if '2.9' in torch_version:
                        if 'cu130' in torch_version:
                            pkg_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nunchaku/nunchaku-1.2.1%2Bcu13.0torch2.9-cp310-cp310-win_amd64.whl'
                            pkg_name = 'nunchaku-1.2.1+cu13.0torch2.9-cp310-cp310-win_amd64.whl'
                            target_ver = '1.2.1+cu13.0torch2.9'
                        else:
                            pkg_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nunchaku/nunchaku-1.2.1%2Bcu12.8torch2.9-cp310-cp310-win_amd64.whl'
                            pkg_name = 'nunchaku-1.2.1+cu12.8torch2.9-cp310-cp310-win_amd64.whl'
                            target_ver = '1.2.1+cu12.8torch2.9'
                    else:
                        pkg_url = 'https://www.modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/nunchaku-1.0.2%2Btorch2.7-cp310-cp310-win_amd64.whl'
                        pkg_name = 'nunchaku-1.0.2+torch2.7-cp310-cp310-win_amd64.whl'
                        target_ver = '1.0.2+torch2.7'
                elif platform.system() == 'Linux' and sys.version_info.major == 3 and sys.version_info.minor == 10:
                    if '2.9' in torch_version:
                        pkg_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nunchaku/nunchaku-1.2.1%2Bcu12.8torch2.9-cp310-cp310-linux_x86_64.whl'
                        pkg_name = 'nunchaku-1.2.1+cu12.8torch2.9-cp310-cp310-linux_x86_64.whl'
                        target_ver = '1.2.1+cu12.8torch2.9'
                    elif '2.7' in torch_version:
                        pkg_url = 'https://www.modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/nunchaku-1.0.2%2Btorch2.7-cp310-cp310-linux_x86_64.whl'
                        pkg_name = 'nunchaku-1.0.2+torch2.7-cp310-cp310-linux_x86_64.whl'
                        target_ver = '1.0.2+torch2.7'

                if pkg_url:
                    pkg_path = os.path.abspath(os.path.join(root, pkg_name))
                    print(f'Preparing to install nunchaku, URL: {pkg_url}')
                    has_update_whl = download_if_updated(pkg_url, pkg_path)
                    # 再次检查是否已安装对应版本，防止重复安装
                    if has_update_whl or (target_ver is not None and not is_installed_version('nunchaku', target_ver)) or (platform.system() == 'Linux' and need_nunchaku_install):
                        run(f'"{python}" -s -m pip install --no-user -U {pkg_path}', f'Install {pkg_path}', custom_env=_make_pip_env(), live=True)
        except Exception as e:
            print(f'Error installing nunchaku: {str(e)}')
            print('Skipping nunchaku installation and continuing...')

        try:
            is_mmcv_installed = is_installed_version('mmcv', '2.1.0')
            if platform.system() == 'Windows':
                if not is_mmcv_installed:
                    mmcv_url = 'https://archive1.piwheels.org/simple/mmcv/mmcv-2.1.0-py2.py3-none-any.whl'
                    mmcv_path = os.path.abspath(os.path.join(root, 'mmcv-2.1.0-py2.py3-none-any.whl'))
                    print('check mmcv for Windows...')
                    has_update_mmcv = download_if_updated(mmcv_url, mmcv_path)
                    if has_update_mmcv or not is_installed_version('mmcv', '2.1.0'):
                        print(f'ready to install {mmcv_path}')
                        run(f'"{python}" -s -m pip install --no-user -U {mmcv_path}', f'Install {mmcv_path}', custom_env=_make_pip_env(), live=True)
            elif platform.system() == 'Linux':
                need_reinstall = not is_mmcv_installed
                if is_mmcv_installed:
                    try:
                        import mmcv
                    except:
                        print("mmcv detected but failed to import. It might be a cross-platform conflict. Reinstalling for Linux...")
                        need_reinstall = True

                if need_reinstall:
                    print('Installing mmcv for Linux...')
                    run(f'"{python}" -s -m pip install --no-user -U mmcv==2.1.0', 'Install mmcv', custom_env=_make_pip_env(), live=True)
        except Exception as e:
            print(f'Error installing mmcv: {str(e)}')
            print('Skipping mmcv installation and continuing...')

        try:
            target_llama_ver = '0.3.30'
            is_llama_installed = is_installed_version('llama_cpp_python', target_llama_ver) or is_installed_version('llama_cpp_python', '0.3.30+cu128.basic')
            need_reinstall = not is_llama_installed

            if is_llama_installed and platform.system() == 'Linux':
                try:
                    import llama_cpp
                except:
                    print("llama_cpp_python detected but failed to import. Reinstalling for Linux...")
                    need_reinstall = True

            if need_reinstall:
                llama_url = None
                if platform.system() == 'Windows':
                    llama_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.30%2Bcu128.basic-cp310-cp310-win_amd64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.30+cu128.basic-cp310-cp310-win_amd64.whl'))
                elif platform.system() == 'Linux' and sys.version_info.major == 3 and sys.version_info.minor == 10:
                    llama_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.30%2Bcu128.basic-cp310-cp310-linux_x86_64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.30+cu128.basic-cp310-cp310-linux_x86_64.whl'))

                if llama_url:
                    print('check llama_cpp_python...')
                    has_update_llama = download_if_updated(llama_url, llama_path)
                    if has_update_llama or not is_installed_version('llama_cpp_python', target_llama_ver) or need_reinstall:
                        print(f'ready to install {llama_path}')
                        run(f'"{python}" -s -m pip install --no-user -U --force-reinstall --no-deps {llama_path}', f'Install {llama_path}', custom_env=_make_pip_env(), live=True)
        except Exception as e:
            print(f'Error installing llama_cpp_python: {str(e)}')
            print('Skipping llama_cpp_python installation and continuing...')

        try:
            ensure_nvidia_vfx_installed()
        except Exception as e:
            print(f'Error installing nvidia-vfx: {str(e)}')
            print('Skipping nvidia-vfx installation and continuing...')

        try:
            ensure_descript_audiotools_installed()
        except Exception as e:
            print(f'Error installing descript_audiotools: {str(e)}')
            print('Skipping descript_audiotools installation and continuing...')

    else:
        logger.info(f'环境缺失必要组件或系统不匹配。请参考SimpAI.cn的安装说明重新部署。')
        logger.info(f'The program running environment lacks necessary components or the system does not match. Please refer to the installation instructions on SimpAI.cn to redeploy.')

    if not is_installed(base_pkg):
        logger.error(f"FATAL ERROR: {base_pkg} is not installed and could not be downloaded/installed.")
        logger.error("程序缺失必要的组件且下载失败，无法继续启动。请检查网络连接并重新启动程序。")
        sys.exit(1)

    from simpleai_base import simpleai_base
    logger.info("Checking ...")
    token = simpleai_base.init_local()
    sysinfo = json.loads(token.get_sysinfo().to_json())
    sysinfo.update(dict(did=token.get_sys_did()))
    logger.info(f'GPU: {sysinfo.get("gpu_name")}, RAM: {sysinfo.get("ram_total")}MB, SWAP: {sysinfo.get("ram_swap")}MB, VRAM: {sysinfo.get("gpu_memory")}MB, DiskFree: {sysinfo.get("disk_free")}MB, CUDA: {sysinfo.get("cuda")}, HOST: {sysinfo.get("host_type")}')
    #print(f'[SimpleAI] root: {sysinfo["root_dir"]}, sys_name: {sysinfo["root_name"]}, dev_name:{sysinfo["host_name"]}')

    cuda_raw = sysinfo.get("cuda", None) if isinstance(sysinfo, dict) else None
    min_cuda_code = 12040
    if torch.__version__ == '2.9.1+cu130':
        min_cuda_code = 13000
    cuda_code = None

    def cuda_code_to_string(code: int) -> str:
        major = code // 1000
        minor = (code % 1000) // 10
        patch = code % 10
        if patch:
            return f"{major}.{minor}.{patch}"
        return f"{major}.{minor}"

    try:
        if isinstance(cuda_raw, (int, float)) and not isinstance(cuda_raw, bool):
            cuda_code = int(cuda_raw)
        elif isinstance(cuda_raw, str):
            s = cuda_raw.strip()
            if s.isdigit():
                cuda_code = int(s)
            else:
                m = re.search(r"\bcu(\d{3})\b", s, flags=re.IGNORECASE)
                if m:
                    cu = int(m.group(1))
                    cuda_code = (cu // 10) * 1000 + (cu % 10) * 10
                else:
                    m = re.search(r"(\d+)\.(\d+)", s)
                    if m:
                        major = int(m.group(1))
                        minor = int(m.group(2))
                        cuda_code = major * 1000 + minor * 10
    except Exception:
        cuda_code = None

    if cuda_code is not None and cuda_code < min_cuda_code:
        cuda_display = cuda_code_to_string(cuda_code)
        min_display = cuda_code_to_string(min_cuda_code)
        min_cu_display = f"cu{(min_cuda_code // 1000) * 10 + ((min_cuda_code % 1000) // 10)}"
        logger.warning(f'CUDA driver/runtime version is too low (CUDA: {cuda_display}). Requires CUDA >= {min_display} ({min_cu_display}). Please update your GPU driver: https://www.nvidia.cn/drivers/')
        logger.warning(f'检测到CUDA驱动/运行时版本过低(CUDA: {cuda_display})。需要CUDA >= {min_display} ({min_cu_display})。请更新显卡驱动否则无法启动：https://www.nvidia.cn/drivers/')

    if (sysinfo.get("ram_total", 0)+sysinfo.get("ram_swap", 0))<65536 and not shared.args.disable_backend:
        logger.info(f'The total virtual memory capacity of the system is too small, which will affect the loading and computing efficiency of the model. Please expand the total virtual memory capacity of the system to be greater than 40G.')
        logger.info(f'系统虚拟内存总容量过小，容易引发后端崩溃，建议扩充系统虚拟内存总容量(RAM+SWAP)大于64G。')
        logger.info(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')

    return token, sysinfo


    #Intel Arc
    #conda install pkg-config libuv
    #python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

def prepare_environment():
    REINSTALL_ALL = False
    TRY_INSTALL_XFORMERS = False

    target_path_win = os.path.join(python_embeded_path, 'Lib/site-packages')

    torch_ver = '2.7.1'
    torchvisio_ver = '0.22.1'
    if is_installed("torch"):
        try:
            current_torch_ver = torch.__version__.split('+')[0]  # 获取主版本号
            target_torch_ver = '2.9.1'
            def _parse_semver3(v: str):
                m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", v or "")
                if not m:
                    return None
                return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            current_parsed = _parse_semver3(current_torch_ver)
            target_parsed = _parse_semver3(target_torch_ver)
            if current_parsed is not None and target_parsed is not None and current_parsed < target_parsed:
                logger.info(f'当前使用的PyTorch版本为{current_torch_ver}，如果你的显卡是RTX20系及以上，可尝试使用一键部署升级为PyTorch {target_torch_ver}获得更好的显存利用效率和速度。')
                logger.info(f'请注意：系统不会自动为您更新PyTorch，在正常使用的情况下，您可以自行选择是否升级。')
        except Exception as e:
            logger.error(f'检测PyTorch版本时发生错误: {str(e)}')
    if shared.sysinfo['gpu_brand'] == 'NVIDIA':
        torch_index_url = "https://download.pytorch.org/whl/cu128"
    elif shared.sysinfo['gpu_brand'] == 'AMD':
        if platform.system() == "Windows":
            #pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
            #pip install torch-directml
            torch_index_url = "https://download.pytorch.org/whl/"
        else:
            torch_index_url = "https://download.pytorch.org/whl/rocm6.3"
            torch_ver = '2.7.1'
            torchvisio_ver = '0.22.1'
    elif shared.sysinfo['gpu_brand'] == 'INTEL':
        torch_index_url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/"
    else:
        torch_index_url = "https://download.pytorch.org/whl/"
    torch_index_url = os.environ.get('TORCH_INDEX_URL', torch_index_url)
    torch_command = os.environ.get('TORCH_COMMAND',
                                f"pip install torch=={torch_ver} torchvision=={torchvisio_ver} --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")
    torch_command += target_path_install
    torch_command += f' -i {index_url} '

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if shared.sysinfo['gpu_brand'] == 'AMD' and platform.system() == "Windows" and not is_installed("torch-directml"):
        run_pip(f"install -U -I --no-deps torch-directml", "torch-directml")

    if not is_installed("torchaudio"):
        torch_command = f'pip install torchaudio=={torch_ver} -i {index_url}'
        run(f'"{python}" -m {torch_command}', "Installing torchaudio", "Couldn't install torchaudio", live=True)

    if TRY_INSTALL_XFORMERS:
        xformers_whl_url_win = 'https://download.pytorch.org/whl/cu128/xformers-0.0.31-cp310-cp310-win_amd64.whl'
        xformers_whl_url_linux = 'https://download.pytorch.org/whl/cu128/xformers-0.0.31-cp310-cp310-manylinux_2_28_x86_64.whl'
        if not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.31')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_whl_url_win}", "xformers 0.0.31", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_whl_url_linux}", "xformers 0.0.31")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        logger.info(f'运行环境中有不匹配的依赖，可能曾经被改动。重新部署程序或咨询交流群1005085136。')
    return

def create_placeholder_files():
    checkpoints_dir = config.paths_checkpoints
    if isinstance(checkpoints_dir, list) and checkpoints_dir:
        checkpoints_dir = checkpoints_dir[0]
    if not os.path.exists(checkpoints_dir):
        try:
            os.makedirs(checkpoints_dir)
            logger.info(f"Created checkpoints directory at {checkpoints_dir}")
        except Exception as e:
            logger.error(f"Failed to create checkpoints directory: {e}")
            return

    safetensors_path = os.path.join(checkpoints_dir, "placeholder.safetensors")
    if not os.path.exists(safetensors_path):
        try:
            with open(safetensors_path, 'w') as f:
                f.write("This is a placeholder file for ComfyUI workflow list.")
            logger.info(f"Created placeholder file: {safetensors_path}")
        except Exception as e:
            logger.error(f"Failed to create safetensors placeholder: {e}")

    gguf_path = os.path.join(checkpoints_dir, "placeholder.gguf")
    if not os.path.exists(gguf_path):
        try:
            with open(gguf_path, 'w') as f:
                f.write("This is a placeholder file for ComfyUI workflow list.")
            logger.info(f"Created placeholder file: {gguf_path}")
        except Exception as e:
            logger.error(f"Failed to create gguf placeholder: {e}")

    loras_dir = config.paths_loras
    if isinstance(loras_dir, list) and loras_dir:
        loras_dir = loras_dir[0]
    if not os.path.exists(loras_dir):
        try:
            os.makedirs(loras_dir)
            logger.info(f"Created loras directory at {loras_dir}")
        except Exception as e:
            logger.error(f"Failed to create loras directory: {e}")
            return

    loras_placeholder_path = os.path.join(loras_dir, "placeholder.safetensors")
    if not os.path.exists(loras_placeholder_path):
        try:
            with open(loras_placeholder_path, 'w') as f:
                f.write("This is a placeholder file for LoRA models.")
            logger.info(f"Created placeholder file: {loras_placeholder_path}")
        except Exception as e:
            logger.error(f"Failed to create loras placeholder: {e}")
def ini_args():
    import args_manager
    if not platform.system() == "Darwin" and args_manager.args.disable_backend:
        args_manager.args.always_cpu = 2
    return args_manager.args


def is_ipynb():
    return True if 'ipykernel' in sys.modules and hasattr(sys, '_jupyter_kernel') else False


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.model_loader import load_file_from_url

    vae_approx_filenames = [
        ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
        ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v4.0.safetensors',
        'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
    ]

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.paths_vae_approx[0], file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if shared.args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not shared.args.always_download_new_model:
        if not os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', default_model)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', alternative_model_name)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('checkpoints', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_embeddings[0], file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('loras', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_vae[0], file_name=file_name)

    return default_model, checkpoint_downloads

def download_required_assets():
    from modules.model_loader import load_file_from_url

    vae_approx_filenames = [
        ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
        ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v4.0.safetensors',
        'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
    ]

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.paths_vae_approx[0], file_name=file_name)

def is_port_available(port, host='127.0.0.1'):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except Exception:
        return False

def find_available_port(start_port=7865, max_attempts=100, suppress_logging=False):
    excluded_ports = {7890, 8187, 8188, 8189, 8190}

    for i in range(max_attempts):
        port = start_port + i
        if port in excluded_ports:
            continue

        host = shared.args.listen if hasattr(shared.args, 'listen') else '127.0.0.1'

        if is_port_available(port, host):
            if not suppress_logging:
                if i > 0:
                    logger.info(f"端口 {start_port} 被占用，自动切换到端口: {port}")
                else:
                    logger.info(f"前端使用端口: {port}")
            return port

    return None

def reset_env_args():
    shared.sysinfo = json.loads(shared.token.get_sysinfo().to_json())
    shared.sysinfo.update(dict(did=shared.token.get_sys_did()))

    if '--location' in sys.argv:
        shared.sysinfo["location"] = args.location

    if shared.sysinfo["location"] == 'CN':
        os.environ['HF_MIRROR'] = 'hf-mirror.com'
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        if '--language' not in sys.argv:
            shared.args.language='cn'

    if '--listen' not in sys.argv:
        if is_ipynb():
            shared.args.listen = '127.0.0.1'
        else:
            from enhanced.simpleai import is_fake_or_suspicious_ip, get_best_local_ip
            local_ip = shared.sysinfo["local_ip"]
            if is_fake_or_suspicious_ip(local_ip):
                best_ip = get_best_local_ip()
                if best_ip != '127.0.0.1':
                    logger.info(f"检测到 Fake IP ({local_ip})，切换到本地 IP: {best_ip}")
                    shared.args.listen = best_ip
                else:
                    shared.args.listen = local_ip
            else:
                shared.args.listen = local_ip
    if '--port' not in sys.argv:
        shared.args.port = shared.sysinfo["local_port"]

    host = shared.args.listen
    if not is_port_available(shared.args.port, host):
        available_port = find_available_port(shared.args.port + 1, suppress_logging=True)
        if available_port:
            logger.info(f"端口 {shared.args.port} 被占用，自动切换到: {available_port}")
            shared.args.port = available_port
    else:
        if '--port' in sys.argv:
            logger.info(f"使用指定的前端端口: {shared.args.port}")
        else:
            logger.info(f"使用默认前端端口: {shared.args.port}")
    if shared.args.node_type and shared.args.node_type != "online":
        shared.sysinfo["local_ip"] = '127.0.0.1'
        shared.args.listen = '127.0.0.1'

    from enhanced.simpleai import reset_simpleai_args
    reset_simpleai_args()

# ready_checker()
shared.args = ini_args()
shared.token, shared.sysinfo = check_base_environment()

prepare_environment()


os.environ["RUST_LOG"] = 'off'
shared.upstream_did = shared.token.get_upstream_did()

shared.upstream_did = '' if shared.args.node_type is not None and shared.args.node_type!='online' else shared.upstream_did
logger.info(f'local_did/本地标识: {shared.token.get_sys_did()}, upstream_did/上游标识: {shared.upstream_did if shared.upstream_did else "no upstream node"}')
logger.info(f'nickname/用户昵称: {shared.token.get_guest_user_context().get_nickname()}, user_did/身份标识: {shared.token.get_guest_did()}')

if shared.args.node_type is not None:
    shared.token.reset_node_mode(shared.args.node_type)

if shared.args.reset_admin is not None:
    shared.token.reset_admin(shared.args.reset_admin)

if shared.args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(shared.args.gpu_device_id)
    logger.info(f"Set device to: {shared.args.gpu_device_id}")

if shared.sysinfo["gpu_memory"]<4000 and not shared.args.disable_backend:
    logger.info(f'The GPU memory capacity of the system is too small to run the latest models such as Flux, SD3m, Kolors, and HyDiT properly, and the Comfyd engine will be automatically disabled.')
    logger.info(f'系统GPU显存容量太小，或是检测不到GPU实际容量，可能是操作系统阻止或需要升级硬件。')
    logger.info(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')
    shared.args.async_cuda_allocation = False
    shared.args.disable_async_cuda_allocation = True
    shared.args.disable_comfyd = True

if shared.args.async_cuda_allocation:
    env_var = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
    if env_var is None:
        env_var = "backend:cudaMallocAsync"
    else:
        env_var += ",backend:cudaMallocAsync"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = env_var

from modules import config
from modules.hash_cache import init_cache
os.environ["U2NET_HOME"] = config.paths_inpaint[0]
os.environ["BERT_HOME"] = config.paths_llms[0]
os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if shared.args.hf_mirror is not None :
    os.environ['HF_MIRROR'] = str(shared.args.hf_mirror)
    logger.info(f"Set hf_mirror to:{shared.args.hf_mirror}")

if config.temp_path_cleanup_on_launch:
    logger.info(f'Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        logger.info("[Cleanup] Cleanup successful")
    else:
        logger.info(f"[Cleanup] Failed to delete content of temp dir.")

pyhash_key = shared.token.get_pyhash_key(fooocus_version.version, comfy_version.version, version.get_simplesdxl_ver())
reset_env_args()
env_ready_code = shared.token.check_ready(fooocus_version.version, comfy_version.version, version.get_simplesdxl_ver(), config.path_models_root)
logger.info(f'Env_ready_code: {env_ready_code}')

if not shared.args.disable_backend:
    try:
        download_required_assets()
    except Exception as e:
        logger.error(f"下载必要资源失将在下次启动重试: {e}")

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)
create_placeholder_files()
from webui import *
