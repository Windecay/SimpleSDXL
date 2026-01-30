import os
import ssl
import sys
import json
import importlib
import packaging.version
import platform
import time
import shared
import fooocus_version
import comfy.comfy_version as comfy_version
import enhanced.version as version
import socket
import logging
import shutil

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

cleanup_obsolete_custom_nodes()

os.environ["SIMPAI_LOG_FILE"] = get_log_file()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["translators_default_region"] = "China"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

def install_package_with_retry(pkg_name, pkg_version=None, description=None):
    """尝试安装包，先使用阿里源，如果失败则尝试使用清华源"""
    desc = description or f'Installing {pkg_name}'
    errdesc = f"Couldn't install {pkg_name}"

    try:
        if pkg_version:
            pkg_command = f'pip install -U {pkg_name}=={pkg_version} -i {index_url}'
        else:
            pkg_command = f'pip install -U {pkg_name} -i {index_url}'

        run(f'"{python}" -m {pkg_command}', desc, errdesc, live=True)
        return True
    except Exception as e:
        logger.warning(f"阿里源安装{pkg_name}失败: {str(e)}")
        logger.info("尝试使用清华源镜像...")

    try:
        if pkg_version:
            pkg_command = f'pip install -U {pkg_name}=={pkg_version} -i {extra_index_url}'
        else:
            pkg_command = f'pip install -U {pkg_name} -i {extra_index_url}'

        run(f'"{python}" -m {pkg_command}', desc, errdesc, live=True)
        return True
    except Exception as e:
        logger.error(f"使用清华源安装{pkg_name}失败: {str(e)}")
        return False
def check_base_environment():
    print(f"{now_string()} Python {sys.version}")
    print(f"{now_string()} Fooocus version: {fooocus_version.version}")
    print(f"{now_string()} Comfyd version: {comfy_version.version}")
    print(f'{now_string()} {version.get_branch()} version: {version.get_simplesdxl_ver()}')
    print(f'{now_string()} ✦ | 兴趣使然的版本 | ✦ by冰華 ✦')

    base_pkg = "simpleai_base"
    ver_required = "0.3.33"
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
                run(f'"{python}" -m pip install {base_path}', f'Install {base_pkg} {ver_required}')
            else:
                version_installed = importlib.metadata.version(base_pkg)
                if REINSTALL_BASE or packaging.version.parse(ver_required) != packaging.version.parse(version_installed):
                    logger.info(f"正在更新 {base_pkg}: {version_installed} -> {ver_required}")
                    run(f'"{python}" -m pip install -U {base_path}', f'Update {base_pkg} {ver_required}')
        else:
            if not is_installed(base_pkg):
                logger.error(f"缺失必要的包 {base_pkg} 且下载失败，程序可能无法正常运行。请检查网络连接并重新启动。")
            else:
                logger.warning(f"无法下载更新包 {base_pkg}，将继续使用当前版本 {importlib.metadata.version(base_pkg)}。")

    if is_installed("sageattention"):
        extra_pkgs = [('comfyui_embedded_docs', 'comfyui_embedded_docs==0.2.3'), ('socketio', 'python-socketio'), ('jsonpatch', 'jsonpatch'), 
                ('alembic', 'alembic'), ('sqlalchemy', 'SQLAlchemy'), ('pyloudnorm', 'pyloudnorm'), ('pydantic', 'pydantic~=2.0'), ('pydantic_settings', 'pydantic-settings~=2.0')]
        for (extra_pkg, extra_pkg_name) in extra_pkgs:
            if not is_installed(extra_pkg):
                pkg_command = f'pip install {extra_pkg_name} -i {index_url}'
                run(f'"{python}" -m {pkg_command}', f'Installing {extra_pkg_name}', f"Couldn't install {extra_pkg_name}", live=True)

        update_pkgs = [('comfyui_frontend_package', '1.37.11'), ('comfyui_workflow_templates', '0.8.24'), ('comfyui-embedded-docs', '0.4.0'), ('transformers', '4.56.2'), ('bitsandbytes', '0.45.5'), ('accelerate', '1.10.1'), ('av', '14.2.0'), ('yarl', '1.18.0'), ('gguf', '0.14.0'),
                       ('sentencepiece', '0.2.0'), ('diffusers', '0.36.0'), ('huggingface_hub', '0.35.1'), ('peft', '0.17.1'), ('tokenizers', '0.22.1'), ('tiktoken', '0.11.0'), ('librosa', '0.11.0'), ('moviepy', '2.2.1'), ('piexif', '1.1.3'), ('deepdiff', '8.6.0'), ('pydantic', '2.12.2'),
                       ('GitPython', '3.1.45'), ('PyGithub', '2.8.1'), ('matrix-nio', '0.24.0'), ('toml', '0.10.2'), ('uv', '0.9.3'), ('clip-interrogator', '0.6.0'), ('simpleeval', '1.0.3'), ('compel', '2.3.0'), ('rotary-embedding-torch', '0.8.9'), ('hydra-core', '1.3.2'), ('uuid7', '0.1.0'), ('aiosqlite', '0.21.0'), ('configs','3.0.3'),
                       ('mmdet', '3.3.0'), ('mmengine', '0.10.7'), ('munkres', '1.1.4'), ('terminaltables', '3.1.10'), ('color-matcher', '0.6.0'), ('natsort', '8.4.0'), ('olefile', '0.47'), ('taichi', '1.7.4'), ('torchdiffeq', '0.2.5'), ('lark', '1.3.1'), ('comfy-kitchen', '0.2.7')]
        for (update_pkg_name, update_pkg_version) in update_pkgs:
            if not is_installed_version(update_pkg_name, update_pkg_version):
                success = install_package_with_retry(update_pkg_name, update_pkg_version)
                if not success:
                    logger.error(f"无法安装{update_pkg_name}，请检查网络状态")
        if not is_installed_version("facenet-pytorch", "2.6.0"):
            logger.info("Installing facenet-pytorch==2.6.0 with --no-deps")
            run_pip(f"install -U facenet-pytorch==2.6.0 --no-deps", "facenet-pytorch==2.6.0")
        try:
            is_torch29_nunchaku = is_installed_version('nunchaku', '1.2.1+torch2.9')
            is_torch27_nunchaku = is_installed_version('nunchaku', '1.2.1+torch2.7')

            need_nunchaku_install = not is_torch29_nunchaku and not is_torch27_nunchaku

            if not need_nunchaku_install and platform.system() == 'Linux':
                try:
                    import nunchaku
                except:
                    print("nunchaku detected but failed to import. It might be a cross-platform conflict. Reinstalling for Linux...")
                    need_nunchaku_install = True

            if need_nunchaku_install:
                import torch
                torch_version = torch.__version__
                print(f'Detected PyTorch version: {torch_version}')

                pkg_url = None
                if platform.system() == 'Windows':
                    if '2.9' in torch_version:
                        pkg_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nunchaku/nunchaku-1.2.1%2Bcu12.8torch2.9-cp310-cp310-win_amd64.whl'
                        pkg_name = 'nunchaku-1.2.1+cu12.8torch2.9-cp310-cp310-win_amd64.whl'
                    else:
                        pkg_url = 'https://www.modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/nunchaku-1.0.2%2Btorch2.7-cp310-cp310-win_amd64.whl'
                        pkg_name = 'nunchaku-1.0.2+torch2.7-cp310-cp310-win_amd64.whl'
                elif platform.system() == 'Linux' and sys.version_info.major == 3 and sys.version_info.minor == 10:
                    if '2.9' in torch_version:
                        pkg_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/nunchaku/nunchaku-1.2.1%2Bcu12.8torch2.9-cp310-cp310-linux_x86_64.whl'
                        pkg_name = 'nunchaku-1.2.1+cu12.8torch2.9-cp310-cp310-linux_x86_64.whl'
                    elif '2.7' in torch_version:
                        pkg_url = 'https://www.modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/nunchaku-1.0.2%2Btorch2.7-cp310-cp310-linux_x86_64.whl'
                        pkg_name = 'nunchaku-1.0.2+torch2.7-cp310-cp310-linux_x86_64.whl'

                if pkg_url:
                    pkg_path = os.path.abspath(os.path.join(root, pkg_name))
                    print(f'Preparing to install nunchaku, URL: {pkg_url}')
                    has_update_whl = download_if_updated(pkg_url, pkg_path)
                    # 再次检查是否已安装对应版本，防止重复安装
                    target_ver = '1.2.1+torch2.9' if '2.9' in torch_version else '1.2.1+torch2.7'
                    if has_update_whl or not is_installed_version('nunchaku', target_ver) or (platform.system() == 'Linux' and need_nunchaku_install):
                        run(f'"{python}" -m pip install -U {pkg_path}', f'Install {pkg_path}', live=True)
        except Exception as e:
            print(f'Error installing nunchaku: {str(e)}')
            print('Skipping nunchaku installation and continuing...')

        try:
            if platform.system() == 'Windows' and not is_installed_version('SAM_2', '1.0'):
                sam_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/sam/SAM_2-1.0-cp310-cp310-win_amd64.whl'
                sam_path = os.path.abspath(os.path.join(root, 'SAM_2-1.0-cp310-cp310-win_amd64.whl'))
                print('check SAM_2...')
                has_update_sam = download_if_updated(sam_url, sam_path)
                is_sam_version_ok = is_installed_version('SAM_2', '1.0')

                if has_update_sam or not is_sam_version_ok:
                    print(f'ready to install {sam_path}')
                    run(f'"{python}" -m pip install -U {sam_path}', f'Install {sam_path}', live=True)
        except Exception as e:
            print(f'Error installing SAM_2: {str(e)}')
            print('Skipping SAM_2 installation and continuing...')

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
                        run(f'"{python}" -m pip install -U {mmcv_path}', f'Install {mmcv_path}', live=True)
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
                    run(f'"{python}" -m pip install -U mmcv==2.1.0', 'Install mmcv', live=True)
        except Exception as e:
            print(f'Error installing mmcv: {str(e)}')
            print('Skipping mmcv installation and continuing...')

        try:
            is_llama_installed = is_installed_version('llama_cpp_python', '0.3.16')
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
                    llama_url = 'https://www.modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.16-cp310-cp310-win_amd64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.16-cp310-cp310-win_amd64.whl'))
                elif platform.system() == 'Linux' and sys.version_info.major == 3 and sys.version_info.minor == 10:
                    llama_url = 'https://modelscope.cn/models/windecay/SimpAI_dev/resolve/master/libs/llama/llama_cpp_python-0.3.16-cp310-cp310-linux_x86_64.whl'
                    llama_path = os.path.abspath(os.path.join(root, 'llama_cpp_python-0.3.16-cp310-cp310-linux_x86_64.whl'))

                if llama_url:
                    print('check llama_cpp_python...')
                    has_update_llama = download_if_updated(llama_url, llama_path)
                    if has_update_llama or not is_installed_version('llama_cpp_python', '0.3.16') or need_reinstall:
                        print(f'ready to install {llama_path}')
                        run(f'"{python}" -m pip install -U --force-reinstall --no-deps {llama_path}', f'Install {llama_path}', live=True)
        except Exception as e:
            print(f'Error installing llama_cpp_python: {str(e)}')
            print('Skipping llama_cpp_python installation and continuing...')

        if platform.system() == 'Windows' and is_installed("rembg") and not is_installed("facexlib") and not is_installed("insightface"):
            logger.info(f'Due to Windows restrictions, The new version of SimpleSDXL requires downloading a new installation package, updating the system environment, and then running it. Download URL: https://hf-mirror.com/metercai/SimpleSDXL2/')
            logger.info(f'受组件安装限制，SimpleSDXL2新版本(增加对混元、可图和SD3支持)需要下载新的程序包和基本模型包。具体操作详见：https://hf-mirror.com/metercai/SimpleSDXL2/')
            logger.info(f'If not updated, you can run the commit version using the following scripte: run_SimpleSDXL_commit.bat')
            logger.info(f'如果不升级，可下载SimpleSDXL1的独立分支完全包(未来仅修bug不加功能): https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleSDXL1_win64_all.exe.7z; 也可点击run_SimpleSDXL_commit.bat继续运行旧版本(历史存档,无法修bug也不加功能)。')
            logger.info(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')
            sys.exit(0)
        if platform.system() == 'Windows' and is_installed("facexlib") and is_installed("insightface") and (not is_installed("cpm_kernels") or not is_installed_version("bitsandbytes", "0.45.5")):
            logger.info(f'运行环境中缺乏必要组件或组件版本不匹配, 或最新程序环境已升级。请参考SimpAI.cn的安装说明重新部署。')
            logger.info(f'The program running environment lacks necessary components, or the latest program environment package has been upgraded. Please refer to the installation instructions on SimpAI.cn to redeploy.')
            logger.info(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')
            sys.exit(0)
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
    logger.info(f'GPU: {sysinfo["gpu_name"]}, RAM: {sysinfo["ram_total"]}MB, SWAP: {sysinfo["ram_swap"]}MB, VRAM: {sysinfo["gpu_memory"]}MB, DiskFree: {sysinfo["disk_free"]}MB, CUDA: {sysinfo["cuda"]}, HOST: {sysinfo["host_type"]}')
    #print(f'[SimpleAI] root: {sysinfo["root_dir"]}, sys_name: {sysinfo["root_name"]}, dev_name:{sysinfo["host_name"]}')

    if (sysinfo["ram_total"]+sysinfo["ram_swap"])<40960 and not shared.args.disable_backend:
        logger.info(f'The total virtual memory capacity of the system is too small, which will affect the loading and computing efficiency of the model. Please expand the total virtual memory capacity of the system to be greater than 40G.')
        logger.info(f'系统虚拟内存总容量过小，会影响模型的加载与计算效率，请扩充系统虚拟内存总容量(RAM+SWAP)大于40G。')
        logger.info(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')
        # sys.exit(0)

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
            import torch
            current_torch_ver = torch.__version__.split('+')[0]  # 获取主版本号
            if current_torch_ver != '2.9.0':
                logger.info(f'当前使用的PyTorch版本为{current_torch_ver}，可尝试使用一键部署升级为PyTorch 2.9.0获得更好的显存利用效率和速度。')
                logger.info(f'请注意：系统不会自动为您更新PyTorch，您可以自行选择是否升级。')
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
        # if len(met_diff.keys())>0:
        #     for p in met_diff.keys():
        #         logger.info(f'Uninstall {p}.{met_diff[p]} ...')
        #         run(f'"{python}" -m pip uninstall -y {p}=={met_diff[p]}')
        # if is_win32_standalone_build:
        #     run_pip(f"install -r \"{requirements_file}\" -t {target_path_win}", "requirements")
        # else:
        #     run_pip(f"install -r \"{requirements_file}\"", "requirements", live=True)
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
    logger.info(f'系统GPU显存容量太小，无法正常运行Flux, SD3m, Kolors和HyDiT等最新模型，将自动禁用Comfyd引擎。请知晓，尽早升级硬件。')
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
#if env_ready_code!=0 and env_ready_code!=4:
#    print("系统环境检测不达标。请根据前面提示信息，重新检查并更新后再启动!")
#    print(f'有任何疑问可到SimpleSDXL的QQ群交流: 1005085136')
#    sys.exit(0)

# if not shared.args.disable_backend:
    # config.default_base_model_name, config.checkpoint_downloads = download_models(
    #     config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    #     config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

    # 检查默认模型是否存在
    # default_model_path = shared.modelsinfo.get_file_path_by_name('diffusion_models', config.default_base_model_name)
    # if not os.path.exists(default_model_path):
    #     logger.error(f"默认模型尚未下载: {config.default_base_model_name}")
    #     logger.error(f"请运行模型检测器或手动下载模型到: {os.path.dirname(default_model_path)}")
    #     # 设置标志以便UI显示错误信息
    #     shared.args.absent_model = True

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)
create_placeholder_files()
from webui import *

