"""
统一的触发词获取模块
整合了 ComfyUI-Lora-Auto-Trigger-Words 的实现逻辑，并保持向后兼容

触发词获取优先级：
1. Civitai API (trainedWords) - 最高优先级，数据最准确
2. 模型元数据 (ss_tag_frequency) - 按训练频率排序
3. 模型元数据 (ss_output_name) - 输出名称
4. 文件名 - 最后的回退选项
"""

import hashlib
import json
import os
import re
from datetime import datetime

import requests

# Civitai触发词缓存文件路径
CIVITAI_CACHE_FILE = "./loras_tags.json"


def calculate_sha256(file_path: str) -> str:
    """
    计算文件的SHA256哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        SHA256哈希字符串
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_json_from_file(file_path: str) -> dict:
    """
    从文件加载JSON数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        JSON数据字典，失败返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"[TriggerWords] JSON解码错误: {file_path}, {e}")
        return None


def save_dict_to_json(data_dict: dict, file_path: str):
    """
    保存字典到JSON文件
    
    Args:
        data_dict: 要保存的数据字典
        file_path: 目标文件路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_dict, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[TriggerWords] 保存JSON失败: {e}")


def get_civitai_model_info(hash_value: str) -> dict:
    """
    通过SHA256哈希从Civitai API获取模型信息
    
    Args:
        hash_value: 模型文件的SHA256哈希值
        
    Returns:
        Civitai API返回的模型信息，失败返回None
    """
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.Timeout:
        print(f"[TriggerWords] Civitai API请求超时")
    except requests.exceptions.RequestException as e:
        print(f"[TriggerWords] Civitai API请求失败: {e}")
    return None


def get_civitai_trigger_words(lora_path: str, lora_name: str, force_fetch: bool = False) -> list:
    """
    从Civitai获取触发词（优先级最高）
    
    实现参考 ComfyUI-Lora-Auto-Trigger-Words 的 load_and_save_tags 函数
    
    Args:
        lora_path: LoRA文件的完整路径
        lora_name: LoRA文件名（不含扩展名）
        force_fetch: 是否强制从Civitai重新获取（忽略缓存）
        
    Returns:
        触发词列表，按Civitai返回的顺序排列
    """
    # 尝试从缓存加载
    cached_tags = load_json_from_file(CIVITAI_CACHE_FILE)
    output_tags = cached_tags.get(lora_name, None) if cached_tags is not None else None
    
    if output_tags is not None and not force_fetch:
        print(f"[TriggerWords] 从缓存获取Civitai触发词: {lora_name}")
        # 处理缓存数据格式
        if isinstance(output_tags, list):
            # 如果列表只有一个元素且包含逗号，说明是旧格式（所有触发词拼接在一起）
            if len(output_tags) == 1 and isinstance(output_tags[0], str) and "," in output_tags[0]:
                # 按逗号分割成多个触发词
                trigger_words = [word.strip() for word in output_tags[0].split(",") if word.strip()]
                print(f"[TriggerWords] 缓存数据格式转换: 从1个元素转换为{len(trigger_words)}个触发词")
                
                # 应用智能过滤，确保第一个触发词是真正的触发词
                non_common_words = [word for word in trigger_words if not _is_common_tag(word)]
                common_words = [word for word in trigger_words if _is_common_tag(word)]
                
                if non_common_words:
                    # 如果有非通用标签，将它们放在前面
                    filtered_words = non_common_words + common_words
                    print(f"[TriggerWords] Civitai智能过滤: 非通用标签{len(non_common_words)}个, 通用标签{len(common_words)}个")
                    print(f"[DEBUG] Civitai前5个非通用标签: {non_common_words[:5]}")
                    return filtered_words
                else:
                    # 如果都是通用标签，保持原顺序
                    return trigger_words
            
            return output_tags
        return []
    
    # 计算SHA256并查询Civitai
    try:
        print(f"[TriggerWords] 计算LoRA哈希值: {lora_name}")
        lora_hash = calculate_sha256(lora_path)
        
        print(f"[TriggerWords] 查询Civitai API...")
        model_info = get_civitai_model_info(lora_hash)
        
        if model_info is not None and "trainedWords" in model_info:
            trained_words = model_info["trainedWords"]
            print(f"[TriggerWords] 从Civitai获取到 {len(trained_words)} 个触发词")
            
            # 保存到缓存
            if cached_tags is None:
                cached_tags = {}
            cached_tags[lora_name] = trained_words
            save_dict_to_json(cached_tags, CIVITAI_CACHE_FILE)
            
            return trained_words
        else:
            print(f"[TriggerWords] Civitai未找到模型信息")
            # 记录空结果，避免重复查询
            if cached_tags is None:
                cached_tags = {}
            cached_tags[lora_name] = []
            save_dict_to_json(cached_tags, CIVITAI_CACHE_FILE)
            
    except Exception as e:
        print(f"[TriggerWords] Civitai获取失败: {e}")
    
    return []


def _is_common_tag(tag: str) -> bool:
    """
    判断是否为常见通用标签（非触发词）
    
    这些标签通常出现在训练数据中，但不是真正的触发词
    """
    # 常见通用标签列表（小写）
    common_tags = {
        # 人物相关
        '1girl', '1boy', '2girls', '2boys', '3girls', '3boys', 'multiple girls', 'multiple boys',
        'solo', 'couple', 'group',
        # 身体特征
        'long hair', 'short hair', 'white hair', 'black hair', 'blonde hair', 'brown hair',
        'red hair', 'blue hair', 'green hair', 'pink hair', 'purple hair', 'silver hair',
        'gradient hair', 'multicolored hair', 'two-tone hair',
        'blue eyes', 'green eyes', 'brown eyes', 'red eyes', 'yellow eyes', 'purple eyes',
        'black eyes', 'white eyes', 'orange eyes', 'pink eyes',
        'small breasts', 'medium breasts', 'large breasts', 'huge breasts',
        # 服装相关
        'dress', 'white dress', 'black dress', 'red dress', 'blue dress', 'green dress',
        'school uniform', 'swimsuit', 'bikini', 'maid', 'nurse', 'waitress',
        'skirt', 'shorts', 'pants', 'jeans', 'shirt', 'blouse', 'jacket', 'coat',
        # 姿势相关
        'standing', 'sitting', 'lying', 'walking', 'running', 'jumping',
        'looking at viewer', 'looking away', 'from behind', 'from side', 'from above', 'from below',
        # 背景相关
        'simple background', 'white background', 'black background', 'grey background',
        'outdoors', 'indoors', 'sky', 'cloud', 'tree', 'flower', 'water', 'building',
        # 画面风格
        'highres', 'absurdres', 'best quality', 'high quality', 'normal quality', 'low quality',
        'masterpiece', 'great quality', 'good quality', 'average quality', 'bad quality',
        'worst quality', 'very bad quality',
        # 其他常见标签
        'smile', 'open mouth', 'closed mouth', 'blush', 'tears', 'sweat',
        'gloves', 'shoes', 'boots', 'socks', 'stockings', 'pantyhose',
        'hat', 'headwear', 'hair ornament', 'ribbon', 'bow', 'flower', 'jewelry',
        'bangs', 'ponytail', 'twintails', 'braid', 'ahoge', 'sidelocks',
        'pointy ears', 'animal ears', 'cat ears', 'dog ears', 'rabbit ears', 'fox ears',
        'tail', 'cat tail', 'dog tail', 'fox tail', 'rabbit tail',
        'wings', 'angel wings', 'demon wings', 'fairy wings',
        'no humans', 'scenery', 'landscape', 'cityscape', 'seascape',
    }
    
    tag_lower = tag.lower().strip()
    return tag_lower in common_tags


def get_metadata_trigger_words(lora_path: str) -> list:
    """
    从safetensors文件元数据获取触发词（按训练频率排序）
    
    实现参考 ComfyUI-Lora-Auto-Trigger-Words 的 get_metadata + sort_tags_by_frequency 函数
    
    Args:
        lora_path: LoRA文件的完整路径
        
    Returns:
        触发词列表，按训练频率降序排列
    """
    if not lora_path.endswith(".safetensors"):
        return []
    
    try:
        with open(lora_path, "rb") as file:
            # 读取header大小
            # https://github.com/huggingface/safetensors#format
            # 8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header
            header_size = int.from_bytes(file.read(8), "little", signed=False)
            
            if header_size <= 0:
                return []
            
            header = file.read(header_size)
            if header is None:
                return []
            
            header_json = json.loads(header)
            metadata = header_json.get("__metadata__", {})
            
            if not metadata:
                return []
            
            # 从 ss_tag_frequency 提取并按频率排序
            if "ss_tag_frequency" in metadata:
                tag_freq_data = metadata["ss_tag_frequency"]
                
                # 如果是字符串，需要解析JSON
                if isinstance(tag_freq_data, str):
                    try:
                        tag_freq_data = json.loads(tag_freq_data)
                    except json.JSONDecodeError:
                        return []
                
                if not isinstance(tag_freq_data, dict):
                    return []
                
                # 统计每个标签的总训练次数
                tag_counts = {}
                for bucket_value in tag_freq_data.values():
                    if isinstance(bucket_value, dict):
                        for tag, count in bucket_value.items():
                            tag = str(tag).strip()
                            if tag:
                                tag_counts[tag] = tag_counts.get(tag, 0) + count
                
                # 按训练次数降序排序
                sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                
                # 智能过滤：优先保留非通用标签
                non_common_tags = []
                common_tags_list = []
                
                for tag, count in sorted_tags:
                    if not _is_common_tag(tag):
                        non_common_tags.append((tag, count))
                    else:
                        common_tags_list.append((tag, count))
                
                # 如果有非通用标签，将它们放在前面
                if non_common_tags:
                    result = [tag for tag, count in non_common_tags] + [tag for tag, count in common_tags_list]
                    print(f"[DEBUG] 智能过滤生效: 非通用标签{len(non_common_tags)}个, 通用标签{len(common_tags_list)}个")
                    print(f"[DEBUG] 前5个非通用标签: {[tag for tag, count in non_common_tags[:5]]}")
                else:
                    result = [tag for tag, count in sorted_tags]
                    print(f"[DEBUG] 智能过滤未生效: 所有标签都是通用标签")
                
                if result:
                    print(f"[TriggerWords] 从元数据获取到 {len(result)} 个触发词(ss_tag_frequency)")
                    print(f"[DEBUG] 第一个触发词: {result[0]}")
                
                return result
            
    except Exception as e:
        print(f"[TriggerWords] 读取元数据失败: {e}")
    
    return []


def get_output_name_trigger_words(lora_path: str) -> str:
    """
    从元数据的 ss_output_name 获取触发词
    
    Args:
        lora_path: LoRA文件的完整路径
        
    Returns:
        输出名称字符串，失败返回空字符串
    """
    if not lora_path.endswith(".safetensors"):
        return ""
    
    try:
        with open(lora_path, "rb") as file:
            header_size = int.from_bytes(file.read(8), "little", signed=False)
            
            if header_size <= 0:
                return ""
            
            header = file.read(header_size)
            if header is None:
                return ""
            
            header_json = json.loads(header)
            metadata = header_json.get("__metadata__", {})
            
            if "ss_output_name" in metadata:
                output_name = metadata["ss_output_name"]
                if output_name and isinstance(output_name, str):
                    output_name = output_name.strip()
                    # 去除常见的版本号后缀 (如 -v1, -v2, _v1, _v2 等)
                    output_name = re.sub(r"[-_]v?[0-9]+$", "", output_name)
                    print(f"[TriggerWords] 从元数据获取触发词(ss_output_name): {output_name}")
                    return output_name
            
    except Exception as e:
        print(f"[TriggerWords] 读取ss_output_name失败: {e}")
    
    return ""


def get_filename_trigger_words(lora_name: str) -> str:
    """
    使用LoRA文件名作为触发词（最后的回退选项）
    
    Args:
        lora_name: LoRA文件名（不含扩展名）
        
    Returns:
        处理后的文件名
    """
    if lora_name and lora_name.strip():
        trigger_word = lora_name.strip()
        # 去除版本号后缀
        trigger_word = re.sub(r"[-_]v?[0-9]+$", "", trigger_word)
        print(f"[TriggerWords] 使用LoRA名称作为触发词: {trigger_word}")
        return trigger_word
    return ""


def get_trigger_words(
    lora_path: str,
    lora_name: str,
    force_fetch_civitai: bool = False,
    include_civitai: bool = True,
    include_metadata: bool = True,
    include_output_name: bool = True,
    include_filename: bool = True
) -> dict:
    """
    获取触发词的统一入口函数
    
    按优先级获取触发词：
    1. Civitai API (trainedWords) - 最高优先级
    2. 模型元数据 (ss_tag_frequency) - 按训练频率排序
    3. 模型元数据 (ss_output_name)
    4. 文件名
    
    Args:
        lora_path: LoRA文件的完整路径
        lora_name: LoRA文件名（不含扩展名）
        force_fetch_civitai: 是否强制从Civitai重新获取
        include_civitai: 是否包含Civitai触发词
        include_metadata: 是否包含元数据触发词
        include_output_name: 是否包含输出名称
        include_filename: 是否包含文件名回退
        
    Returns:
        {
            "civitai_words": [],      # Civitai触发词列表
            "metadata_words": [],     # 元数据触发词列表（按频率排序）
            "output_name": "",        # 输出名称
            "filename_word": "",      # 文件名触发词
            "first_trigger_word": "", # 第一个触发词（用于快速获取）
            "all_trigger_words": [],  # 所有触发词列表（去重合并）
            "source": ""              # 第一个触发词的来源
        }
    """
    result = {
        "civitai_words": [],
        "metadata_words": [],
        "output_name": "",
        "filename_word": "",
        "first_trigger_word": "",
        "all_trigger_words": [],
        "source": ""
    }
    
    # 按优先级获取触发词，一旦找到有效数据就不再执行后续方式
    # 1. Civitai API (最高优先级)
    if include_civitai:
        civitai_words = get_civitai_trigger_words(lora_path, lora_name, force_fetch_civitai)
        result["civitai_words"] = civitai_words
        if civitai_words:
            # Civitai有数据，直接返回，不再执行后续方式
            result["first_trigger_word"] = civitai_words[0]
            result["source"] = "civitai"
            result["all_trigger_words"] = civitai_words
            return result
    
    # 2. 元数据 ss_tag_frequency
    if include_metadata:
        metadata_words = get_metadata_trigger_words(lora_path)
        result["metadata_words"] = metadata_words
        if metadata_words:
            # 元数据有数据，直接返回，不再执行后续方式
            result["first_trigger_word"] = metadata_words[0]
            result["source"] = "metadata"
            result["all_trigger_words"] = metadata_words
            return result
    
    # 3. 元数据 ss_output_name
    if include_output_name:
        output_name = get_output_name_trigger_words(lora_path)
        result["output_name"] = output_name
        if output_name:
            # 输出名称有数据，直接返回，不再执行后续方式
            result["first_trigger_word"] = output_name
            result["source"] = "output_name"
            result["all_trigger_words"] = [output_name]
            return result
    
    # 4. 文件名回退
    if include_filename:
        filename_word = get_filename_trigger_words(lora_name)
        result["filename_word"] = filename_word
        if filename_word:
            # 文件名有数据，直接返回
            result["first_trigger_word"] = filename_word
            result["source"] = "filename"
            result["all_trigger_words"] = [filename_word]
            return result
    
    # 所有方式都没有找到数据，返回空结果
    return result


def get_first_trigger_word(
    lora_path: str,
    lora_name: str,
    force_fetch_civitai: bool = False
) -> str:
    """
    获取第一个触发词（简化接口）
    
    用于节点执行时快速获取触发词
    
    Args:
        lora_path: LoRA文件的完整路径
        lora_name: LoRA文件名（不含扩展名）
        force_fetch_civitai: 是否强制从Civitai重新获取
        
    Returns:
        第一个触发词字符串
    """
    result = get_trigger_words(
        lora_path,
        lora_name,
        force_fetch_civitai=force_fetch_civitai,
        include_civitai=True,
        include_metadata=True,
        include_output_name=True,
        include_filename=True
    )
    print(f"[DEBUG] get_trigger_words返回的result: {result}")
    print(f"[DEBUG] first_trigger_word: {result['first_trigger_word']}")
    print(f"[DEBUG] source: {result['source']}")
    print(f"[DEBUG] civitai_words: {result['civitai_words']}")
    print(f"[DEBUG] metadata_words前5个: {result['metadata_words'][:5] if result['metadata_words'] else []}")
    return result["first_trigger_word"]
