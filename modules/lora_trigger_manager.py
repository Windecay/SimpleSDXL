import os
import csv
import json
from pathlib import Path
import logging
import modules.config
logger = logging.getLogger(__name__)
# 存储触发词数据的CSV文件路径
if modules.config.paths_loras and len(modules.config.paths_loras) > 0:
    # 确保data目录存在
    lora_data_dir = os.path.join(modules.config.paths_loras[0], 'data')
    if not os.path.exists(lora_data_dir):
        os.makedirs(lora_data_dir, exist_ok=True)
    # 设置触发词文件路径
    TRIGGER_WORDS_FILE = os.path.join(lora_data_dir, 'lora_trigger_words.csv')
else:
    # 如果没有配置LoRA路径，使用默认位置
    TRIGGER_WORDS_FILE = os.path.join('..', '..', 'SimpleModels', 'loras', 'data', 'lora_trigger_words.csv')

class LoRATriggerManager:
    """管理LoRA触发词的类"""
    
    def __init__(self):
        # 确保数据目录存在
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.trigger_words = {}
        self.load_trigger_words()
    
    def load_trigger_words(self):
        """从CSV文件加载触发词数据"""
        self.trigger_words = {}
        
        # 检查文件是否存在
        if not os.path.exists(TRIGGER_WORDS_FILE):
            return
        
        try:
            with open(TRIGGER_WORDS_FILE, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lora_name = row.get('lora_name', '').strip()
                    trigger_word = row.get('trigger_word', '').strip()
                    if lora_name:
                        self.trigger_words[lora_name] = trigger_word
        except Exception as e:
            print(f"加载LoRA触发词时出错: {e}")
    
    def save_trigger_words(self):
        """保存触发词数据到CSV文件"""
        try:
            with open(TRIGGER_WORDS_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['lora_name', 'trigger_word']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for lora_name, trigger_word in self.trigger_words.items():
                    writer.writerow({
                        'lora_name': lora_name,
                        'trigger_word': trigger_word
                    })
            return True
        except Exception as e:
            print(f"保存LoRA触发词时出错: {e}")
            return False
    
    def get_trigger_word(self, lora_name):
        """获取指定LoRA模型的触发词"""
        return self.trigger_words.get(lora_name, '')
    
    def set_trigger_word(self, lora_name, trigger_word):
        """设置指定LoRA模型的触发词"""
        self.trigger_words[lora_name] = trigger_word.strip()
        return self.save_trigger_words()
    
    def delete_trigger_word(self, lora_name):
        """删除指定LoRA模型的触发词"""
        if lora_name in self.trigger_words:
            del self.trigger_words[lora_name]
            return self.save_trigger_words()
        return True

# 创建全局实例供应用程序使用
trigger_manager = LoRATriggerManager()

# 导出常用函数以便直接调用
def get_lora_trigger_word(lora_name):
    """获取指定LoRA模型的触发词"""
    # 每次调用都重新加载触发词数据，确保获取最新值
    trigger_manager.load_trigger_words()
    return trigger_manager.get_trigger_word(lora_name)


def set_lora_trigger_word(lora_name, trigger_word):
    """设置指定LoRA模型的触发词"""
    return trigger_manager.set_trigger_word(lora_name, trigger_word)


def save_lora_trigger_words():
    """保存所有LoRA触发词"""
    return trigger_manager.save_trigger_words()

# 新增：LoRA UI交互相关函数
def update_trigger_word(lora_model_name):
    """实现LoRA模型变化时加载对应触发词的功能"""
    trigger_word = get_lora_trigger_word(lora_model_name) if lora_model_name != 'None' else ''
    return trigger_word


def save_trigger_word(lora_model_name, trigger_word):
    """实现保存触发词的功能"""
    if lora_model_name != 'None':
        set_lora_trigger_word(lora_model_name, trigger_word)
        logger.info(f"保存触发词 {trigger_word} 到模型 {lora_model_name}")
    return trigger_word


def send_trigger_to_prompt(lora_model_name, trigger_word):
    """实现发送触发词到提示词的功能"""
    import gradio as gr  # 导入需要的gradio模块
    if lora_model_name != 'None' and trigger_word:
        return gr.update()
    return gr.update()