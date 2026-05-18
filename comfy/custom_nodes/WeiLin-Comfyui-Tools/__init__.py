import os

import nodes

# ComfyUI 2.0兼容性：使用nodes.EXTENSION_WEB_DIRS注册JavaScript目录
# 必须在其他导入之前设置
custom_node_dir = os.path.dirname(os.path.realpath(__file__))
js_dir = os.path.join(custom_node_dir, "js_node")
nodes.EXTENSION_WEB_DIRS["weilin-comfyui-tools"] = js_dir


# Server Init
from .install_request import *

install_requirements()


import json
import locale
import logging
import re
import shutil

import comfy.lora
import comfy.utils

from .app.server.prompt_server import *

# 检测系统语言
localLan = locale.getdefaultlocale()[0]
placeholder_text = ""
retrun_name_text = ""
retrun_type_text = ""
node_name_text = ""
node_model_text = ""
placeholder_node_text = ""
placeholder_lora_text = ""
if localLan == "zh_CN":
    placeholder_text = "输入提示词"
    placeholder_lora_text = "Lora信息框"
    placeholder_node_text = "输入节点命名"
    retrun_name_text = "条件"
    retrun_type_text = "条件"
    node_name_text = "WeiLin-Tools-节点工具"
    node_model_text = "模型"
else:
    placeholder_text = "input prompt words"
    placeholder_lora_text = "Lora info box"
    retrun_name_text = "CONDITIONING"
    retrun_type_text = "CONDITIONING"
    node_name_text = "WeiLin Node Tools"
    node_model_text = "MODEL"
    placeholder_node_text = "input node name"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True


def validate_conditioning_output(conditioning_data):
    """
    验证CONDITIONING输出格式是否正确。
    CONDITIONING应该是 [[cond_tensor, dict]] 或 None 的格式。
    这个函数用于确保输出数据结构正确，防止连接到错误节点时引发问题。

    Args:
        conditioning_data: CONDITIONING输出数据

    Returns:
        bool: True表示格式正确，False表示格式错误
    """
    if conditioning_data is None:
        return True

    # 检查是否是列表格式
    if not isinstance(conditioning_data, list):
        print(
            f"[WARNING] CONDITIONING输出格式错误: 期望list，得到{type(conditioning_data)}"
        )
        return False

    # 检查是否至少有一个元素
    if len(conditioning_data) == 0:
        print("[WARNING] CONDITIONING输出为空列表")
        return True

    # 检查第一个元素是否是 [cond, dict] 格式
    first_item = conditioning_data[0]
    if not isinstance(first_item, list) or len(first_item) < 2:
        print(
            f"[WARNING] CONDITIONING内部格式错误: 期望[cond, dict]，得到{type(first_item)}"
        )
        return False

    return True


def get_lora_trigger_words(lora_path, lora_name, force_fetch_civitai=False):
    """
    从LoRA文件中获取触发词（优化版）
    
    整合了 ComfyUI-Lora-Auto-Trigger-Words 的实现逻辑
    
    优先级：
    1. Civitai API (trainedWords) - 最高优先级，数据最准确
    2. 从模型元数据中提取训练词 (ss_tag_frequency) - 按训练频率排序
    3. 从模型元数据中获取输出名称 (ss_output_name)
    4. 使用LoRA文件名作为触发词
    
    Args:
        lora_path: LoRA文件的完整路径
        lora_name: LoRA文件名（不含扩展名）
        force_fetch_civitai: 是否强制从Civitai重新获取（忽略缓存）
        
    Returns:
        第一个触发词字符串
    """
    try:
        from .app.server.prompt_api.trigger_words import get_first_trigger_word
        result = get_first_trigger_word(lora_path, lora_name, force_fetch_civitai)
        print(f"[DEBUG] get_first_trigger_word返回: {result}")
        return result
    except ImportError as e:
        print(f"[TriggerWords] 导入模块失败，使用回退逻辑: {e}")
        # 回退到原有逻辑（向后兼容）
        return _get_lora_trigger_words_fallback(lora_path, lora_name)


def _get_lora_trigger_words_fallback(lora_path, lora_name):
    """
    触发词获取的回退逻辑（向后兼容）
    
    当新模块导入失败时使用此函数
    """
    trigger_words = ""

    try:
        # 尝试从safetensors文件头读取元数据
        if lora_path.endswith(".safetensors"):
            with open(lora_path, "rb") as file:
                # 读取header大小
                header_size = int.from_bytes(file.read(8), "little", signed=False)
                if header_size > 0:
                    header = file.read(header_size)
                    header_json = json.loads(header)
                    metadata = header_json.get("__metadata__", {})

                    # 优先级1: 从ss_tag_frequency提取第一个训练词
                    if "ss_tag_frequency" in metadata:
                        tag_freq = metadata["ss_tag_frequency"]
                        if isinstance(tag_freq, dict):
                            # 遍历所有bucket，找到第一个标签
                            for bucket_value in tag_freq.values():
                                if isinstance(bucket_value, dict):
                                    for tag, count in bucket_value.items():
                                        if tag and tag.strip():
                                            trigger_words = tag.strip()
                                            print(
                                                f"从元数据获取触发词(ss_tag_frequency): {trigger_words}"
                                            )
                                            return trigger_words

                    # 优先级2: 从ss_output_name获取
                    if "ss_output_name" in metadata:
                        output_name = metadata["ss_output_name"]
                        if output_name and output_name.strip():
                            trigger_words = output_name.strip()
                            # 去除常见的版本号后缀 (如 -v1, -v2, _v1, _v2 等)
                            # 版本号格式: -v数字, _v数字, -数字, _数字
                            trigger_words = re.sub(r"[-_]v?[0-9]+$", "", trigger_words)
                            print(
                                f"从元数据获取触发词(ss_output_name): {trigger_words}"
                            )
                            return trigger_words

    except Exception as e:
        print(f"读取LoRA元数据失败: {e}")

    # 优先级3: 使用LoRA文件名作为触发词
    if lora_name and lora_name.strip():
        trigger_words = lora_name.strip()
        # 去除版本号后缀
        trigger_words = re.sub(r"[-_]v?[0-9]+$", "", trigger_words)
        print(f"使用LoRA名称作为触发词: {trigger_words}")

    return trigger_words


class AnyType(str):
    """
    A class representing any type in ComfyUI nodes.
    Used for parameters that can accept any type of input.
    """

    def __ne__(self, __value: object) -> bool:
        return False

    @classmethod
    def INPUT_TYPE(cls):
        return (ANY, {})


ANY = AnyType("*")

# 提示词UI


class WeiLinPromptUI:
    def __init__(self):
        self.loaded_loraA = None

    @classmethod
    def IS_CHANGED(self, auto_random, **kwargs):
        if auto_random:
            return float("nan")

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": placeholder_text,
                    },
                ),
                "auto_random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": placeholder_lora_text,
                    },
                ),
                "temp_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "temp prompt words",
                    },
                ),
                "temp_lora_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "temp prompt words",
                    },
                ),
                "random_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "random template path name",
                    },
                ),
                "opt_text": (ANY, {"default": ""}),
                "opt_clip": ("CLIP",),
                "opt_model": ("MODEL",),
            },
        }

    # RETURN_TYPES = ("STRING",)
    # RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_TYPES = (
        "STRING",
        "CONDITIONING",
        "CLIP",
        "MODEL",
    )
    RETURN_NAMES = (
        "STRING",
        "CONDITIONING",
        "CLIP",
        "MODEL",
    )

    # FUNCTION = "encode"
    FUNCTION = "load_lora_ing"

    OUTPUT_NODE = True

    CATEGORY = node_name_text

    # 加载Lora
    def load_lora_ing(
        self,
        positive="",
        auto_random=False,
        lora_str="",
        temp_str="",
        temp_lora_str="",
        random_template="",
        opt_text="",
        opt_clip=None,
        opt_model=None,
    ):

        model_lora_secondA = opt_model
        clip_lora_secondA = opt_clip

        text_dec = ""
        lora_list = None

        if is_json(positive):
            json_object = json.loads(positive)
            lora_list = json_object.get("lora", None)
            if len(opt_text) > 0:
                text_dec = opt_text + ", " + json_object.get("prompt", "")
            else:
                text_dec = json_object.get("prompt", "")
        else:
            if len(opt_text) > 0:
                text_dec = opt_text + ", " + positive
            else:
                text_dec = positive
        if len(lora_str) > 0:
            json_object = json.loads(lora_str)
            lora_list = json_object

        if auto_random:
            if len(random_template) > 0:
                # 随机Tag获取
                random_tag = go_run_node_auto_random_tag(random_template)
                if len(random_tag["random_tags"]) > 0:
                    positive = random_tag["random_tags"]
                    self.positive = positive
                    if len(opt_text) > 0:
                        text_dec = opt_text + ", " + positive
                    else:
                        text_dec = positive

        # 匹配两种格式：
        # 旧格式(3参数): <wlr:name:model_weight:clip_weight>
        # 新格式(4参数): <wlr:name:model_weight:clip_weight:trigger_weight>
        # 注意：必须先匹配新格式，再匹配旧格式，避免旧格式误匹配新格式标签
        wlr_pattern_new = r"<wlr:([^:]+):([^:]+):([^:]+):([^>]+)>"
        wlr_pattern_old = r"<wlr:([^:]+):([^:]+):([^:>]+)>"

        wlr_matches_new = re.findall(wlr_pattern_new, text_dec)

        # 从文本中移除新格式标签，再匹配旧格式，避免重复匹配
        text_dec_without_new = re.sub(wlr_pattern_new, "", text_dec)
        wlr_matches_old = re.findall(wlr_pattern_old, text_dec_without_new)

        # 如果找到了wlr标签，创建lora列表
        if wlr_matches_new or wlr_matches_old:
            extracted_loras = []

            # 处理新格式(4参数)
            for lora_path, model_weight, text_weight, trigger_weight in wlr_matches_new:
                extracted_loras.append(
                    {
                        "lora": lora_path.strip() + ".safetensors",
                        "weight": float(model_weight.strip()),
                        "text_encoder_weight": float(text_weight.strip()),
                        "trigger_weight": float(trigger_weight.strip()),
                    }
                )

            # 处理旧格式(3参数)
            for lora_path, model_weight, text_weight in wlr_matches_old:
                extracted_loras.append(
                    {
                        "lora": lora_path.strip() + ".safetensors",
                        "weight": float(model_weight.strip()),
                        "text_encoder_weight": float(text_weight.strip()),
                        "trigger_weight": float(
                            text_weight.strip()
                        ),  # 旧格式使用 clip_weight 作为 trigger_weight
                    }
                )

            # 从text_dec中移除这些标签
            clean_text_dec = re.sub(wlr_pattern_new, "", text_dec)
            clean_text_dec = re.sub(wlr_pattern_old, "", clean_text_dec)
            # 清理连续的逗号
            clean_text_dec = re.sub(r",\s*,", ",", clean_text_dec)
            # 清理开头和结尾的逗号
            clean_text_dec = clean_text_dec.strip().strip(",").strip()
            text_dec = clean_text_dec

            # 如果已经有lora_list，需要去重合并
            # 避免同一个Lora被重复处理（来自JSON的lora字段和prompt中的<wlr:...>标签）
            if lora_list is not None:
                # 获取已存在的lora名称集合
                existing_lora_names = {
                    item["lora"].replace(".safetensors", "") for item in lora_list
                }
                # 只添加不存在的lora
                for extracted in extracted_loras:
                    extracted_name = extracted["lora"].replace(".safetensors", "")
                    if extracted_name not in existing_lora_names:
                        lora_list.append(extracted)
            else:
                lora_list = extracted_loras

        # 当模型不为空时
        if opt_model is not None and lora_list is not None:
            # 收集所有触发词
            all_trigger_words = []

            for str_lora_item in lora_list:
                # print(loar_sim_path,str_n_arr)
                strength_model = float(str_lora_item["weight"])
                strength_clip = float(str_lora_item["text_encoder_weight"])
                # 获取触发词权重（用于提示词中的权重标记）
                trigger_weight = float(
                    str_lora_item.get("trigger_weight", strength_clip)
                )

                print(
                    "模型权重strength_model：",
                    strength_model,
                    "CLIP权重strength_clip：",
                    strength_clip,
                    "触发词权重trigger_weight：",
                    trigger_weight,
                )

                lora_path = folder_paths.get_full_path("loras", str_lora_item["lora"])
                if lora_path is None:
                    raise ValueError(f"无法找到Lora文件: {str_lora_item['lora']}")
                print("加载Lora lora_path:", lora_path)

                # 获取触发词
                lora_name = os.path.splitext(str_lora_item["lora"])[0]
                trigger_word = get_lora_trigger_words(lora_path, lora_name)
                print(f"[DEBUG] get_lora_trigger_words返回类型: {type(trigger_word)}")
                print(f"[DEBUG] get_lora_trigger_words返回值: {trigger_word}")
                print(f"[DEBUG] 触发词长度: {len(trigger_word) if trigger_word else 0}")
                print(f"触发词: {trigger_word if trigger_word else '无'}")

                # 收集触发词（带权重）
                if trigger_word:
                    # 格式: 触发词:触发词权重
                    trigger_with_weight = f"{trigger_word}:{trigger_weight}"
                    all_trigger_words.append(trigger_with_weight)

                lora = None
                if self.loaded_loraA is not None:
                    if self.loaded_loraA[0] == lora_path:
                        lora = self.loaded_loraA[1]
                    else:
                        temp = self.loaded_loraA
                        self.loaded_loraA = None
                        del temp

                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loraA = (lora_path, lora)

                model_lora_secondA, clip_lora_secondA = load_lora_for_models(
                    model_lora_secondA,
                    clip_lora_secondA,
                    lora,
                    strength_model,
                    strength_clip,
                )

            # 将所有触发词添加到提示词开头
            if all_trigger_words:
                trigger_text = ", ".join(all_trigger_words) + ", "
                text_dec = trigger_text + text_dec
                print(f"添加触发词到提示词开头: {trigger_text}")

        if opt_clip is not None:
            try:
                tokensA = clip_lora_secondA.tokenize(text_dec)
                outputA = clip_lora_secondA.encode_from_tokens(
                    tokensA, return_pooled=True, return_dict=True
                )
                condA = outputA.pop("cond")
                conditioning_output = [[condA, outputA]]

                # 验证CONDITIONING输出格式
                if not validate_conditioning_output(conditioning_output):
                    print("[ERROR] CONDITIONING输出格式验证失败，返回None")
                    conditioning_output = None

                if auto_random:
                    return {
                        "ui": {"positive": [str(positive)]},
                        "result": (
                            text_dec,
                            conditioning_output,
                            clip_lora_secondA,
                            model_lora_secondA,
                        ),
                    }
                return (
                    text_dec,
                    conditioning_output,
                    clip_lora_secondA,
                    model_lora_secondA,
                )
            except Exception as e:
                print(f"[ERROR] CONDITIONING编码失败: {e}")
                if auto_random:
                    return {
                        "ui": {"positive": [str(positive)]},
                        "result": (
                            text_dec,
                            None,
                            clip_lora_secondA,
                            model_lora_secondA,
                        ),
                    }
                return (
                    text_dec,
                    None,
                    clip_lora_secondA,
                    model_lora_secondA,
                )

        # 没有CLIP输入时，返回None作为CONDITIONING
        if auto_random:
            return {
                "ui": {"positive": [str(positive)]},
                "result": (
                    text_dec,
                    None,
                    clip_lora_secondA,
                    model_lora_secondA,
                ),
            }
        return (
            text_dec,
            None,
            clip_lora_secondA,
            model_lora_secondA,
        )
        # return (model_lora_second, clip_lora_second)


# 提示词UI - 仅使用Lora堆
class WeiLinPromptUIOnlyLoraStack:
    def __init__(self):
        self.loaded_loraA = None

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
            },
            "optional": {
                "lora_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "lora info box",
                    },
                ),
                "temp_lora_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "temp prompt words",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "CLIP",
        "MODEL",
    )
    RETURN_NAMES = (
        "CLIP",
        "MODEL",
    )

    # FUNCTION = "encode"
    FUNCTION = "load_lora_ing"

    # OUTPUT_NODE = False

    CATEGORY = node_name_text

    # 加载Lora
    def load_lora_ing(self, clip=None, model=None, lora_str="", temp_lora_str=""):
        model_lora_secondA = model
        clip_lora_secondA = clip

        lora_list = None

        if len(lora_str) > 0:
            json_object = json.loads(lora_str)
            lora_list = json_object

        # 当模型不为空时
        if model is not None and lora_list is not None:
            for str_lora_item in lora_list:
                # print(loar_sim_path,str_n_arr)
                strength_model = float(str_lora_item["weight"])
                strength_clip = float(str_lora_item["text_encoder_weight"])
                print(
                    "模型权重strength_model：",
                    strength_model,
                    "文本权重strength_clip：",
                    strength_clip,
                )

                lora_path = folder_paths.get_full_path("loras", str_lora_item["lora"])
                if lora_path is None:
                    raise ValueError(f"无法找到Lora文件: {str_lora_item['lora']}")
                print("加载Lora lora_path:", lora_path)
                lora = None
                if self.loaded_loraA is not None:
                    if self.loaded_loraA[0] == lora_path:
                        lora = self.loaded_loraA[1]
                    else:
                        temp = self.loaded_loraA
                        self.loaded_loraA = None
                        del temp

                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loraA = (lora_path, lora)

                model_lora_secondA, clip_lora_secondA = load_lora_for_models(
                    model_lora_secondA,
                    clip_lora_secondA,
                    lora,
                    strength_model,
                    strength_clip,
                )
            else:
                print("Lora堆没有可用的Lora信息")

        return (clip_lora_secondA, model_lora_secondA)
        # return (model_lora_second, clip_lora_second)


# 提示词UI - 不加载Lora


class WeiLinPromptUIWithoutLora:
    def __init__(self):
        pass

    @classmethod
    def IS_CHANGED(self, auto_random, **kwargs):
        if auto_random:
            return float("nan")

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": placeholder_text,
                    },
                ),
                "auto_random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "temp_str": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "temp prompt words",
                    },
                ),
                "random_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "random template path name",
                    },
                ),
                "opt_text": (ANY, {"default": ""}),
                "opt_clip": ("CLIP",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (
        "STRING",
        "CONDITIONING",
        "CLIP",
    )
    RETURN_NAMES = (
        "STRING",
        "CONDITIONING",
        "CLIP",
    )
    FUNCTION = "encode"
    OUTPUT_NODE = True

    CATEGORY = node_name_text

    def encode(
        self,
        positive="",
        auto_random=False,
        temp_str="",
        random_template="",
        opt_text="",
        opt_clip=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        text_dec = ""
        if is_json(positive):
            json_object = json.loads(positive)
            if len(opt_text) > 0:
                text_dec = opt_text + ", " + json_object.get("prompt", "")
            else:
                text_dec = json_object.get("prompt", "")
        else:
            if len(opt_text) > 0:
                text_dec = opt_text + ", " + positive
            else:
                text_dec = positive

        if auto_random:
            if len(random_template) > 0:
                # 随机Tag获取
                random_tag = go_run_node_auto_random_tag(random_template)
                if len(random_tag["random_tags"]) > 0:
                    positive = random_tag["random_tags"]
                    if len(opt_text) > 0:
                        text_dec = opt_text + ", " + positive
                    else:
                        text_dec = positive

        if opt_clip is not None:
            try:
                tokens = opt_clip.tokenize(text_dec)
                conditioning_output = opt_clip.encode_from_tokens_scheduled(tokens)

                # 验证CONDITIONING输出格式
                if not validate_conditioning_output(conditioning_output):
                    print("[ERROR] CONDITIONING输出格式验证失败，返回None")
                    conditioning_output = None

                if auto_random:
                    return {
                        "ui": {"positive": [str(positive)]},
                        "result": (
                            text_dec,
                            conditioning_output,
                            opt_clip,
                        ),
                    }
                return (
                    text_dec,
                    conditioning_output,
                    opt_clip,
                )
            except Exception as e:
                print(f"[ERROR] CONDITIONING编码失败: {e}")
                if auto_random:
                    return {
                        "ui": {"positive": [str(positive)]},
                        "result": (
                            text_dec,
                            None,
                            opt_clip,
                        ),
                    }
                return (
                    text_dec,
                    None,
                    opt_clip,
                )
        if auto_random:
            return {
                "ui": {"positive": [str(positive)]},
                "result": (
                    text_dec,
                    None,
                    opt_clip,
                ),
            }
        return (
            text_dec,
            None,
            opt_clip,
        )


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = comfy.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning(f"NOT LOADED {x}")

    return (new_modelpatcher, new_clip)


def copy_folder(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for item in os.listdir(source_folder):
        source = os.path.join(source_folder, item)
        destination = os.path.join(destination_folder, item)

        if os.path.isdir(source):
            copy_folder(source, destination)
        else:
            shutil.copy2(source, destination)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "WeiLinPromptUI": WeiLinPromptUI,
    "WeiLinPromptUIWithoutLora": WeiLinPromptUIWithoutLora,
    "WeiLinPromptUIOnlyLoraStack": WeiLinPromptUIOnlyLoraStack,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {}

if localLan == "zh_CN":
    NODE_DISPLAY_NAME_MAPPINGS = {
        "WeiLinPromptUI": "WeiLin 全能提示词编辑器",
        "WeiLinPromptUIWithoutLora": "WeiLin 提示词编辑器",
        "WeiLinPromptUIOnlyLoraStack": "WeiLin Lora堆",
    }
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "WeiLinPromptUI": "All-Round WeiLin Prompt Editor",
        "WeiLinPromptUIWithoutLora": "WeiLin Prompt Editor",
        "WeiLinPromptUIOnlyLoraStack": "WeiLin Lora Stack",
    }

WEB_DIRECTORY = "./js_node"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
