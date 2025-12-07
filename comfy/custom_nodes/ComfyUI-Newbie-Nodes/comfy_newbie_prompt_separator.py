"""
NewBie Prompt Separator Node for ComfyUI
分离system prompt和正常prompt，为NewBie CLIP优化
"""

import re
from typing import Tuple, List, Optional


class NewBiePromptSeparator:
    """
    NewBie提示词分离器
    分离system prompt和正常prompt，Jina CLIP不需要system prompt
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "Input text containing both system and user prompts"
                }),
            },
            "optional": {
                "system_prefix": ("STRING", {
                    "default": "System:",
                    "description": "Prefix to identify system prompts"
                }),
                "user_prefix": ("STRING", {
                    "default": "User:",
                    "description": "Prefix to identify user prompts"
                }),
                "auto_detect": ("BOOLEAN", {
                    "default": True,
                    "description": "Auto detect system/user prompts without prefixes"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gemma_prompt", "jina_prompt", "combined_prompt")
    FUNCTION = "separate_prompts"
    CATEGORY = "conditioning"
    TITLE = "NewBie Prompt Separator"

    def separate_prompts(
        self, 
        text: str,
        system_prefix: str = "System:",
        user_prefix: str = "User:",
        auto_detect: bool = True
    ) -> Tuple[str, str, str]:
        """
        分离system prompt和正常prompt
        
        Args:
            text: 输入文本
            system_prefix: system prompt前缀
            user_prefix: user prompt前缀
            auto_detect: 是否自动检测
            
        Returns:
            (gemma_prompt, jina_prompt, combined_prompt)
            - gemma_prompt: Gemma3-4B-IT使用的完整提示（包含system+user）
            - jina_prompt: Jina CLIP使用的提示（仅user部分）
            - combined_prompt: 原始组合提示
        """
        
        if not text.strip():
            return "", "", ""
        
        system_parts = []
        user_parts = []
        
        if auto_detect:
            # 自动检测模式
            lines = text.split('\n')
            current_mode = 'user'  # 默认为用户提示
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检测是否为system prompt
                if self._is_system_prompt(line):
                    current_mode = 'system'
                    # 移除system前缀
                    cleaned_line = re.sub(r'^(system|sys|instruction|instruct):\s*', '', line, flags=re.IGNORECASE)
                    if cleaned_line:
                        system_parts.append(cleaned_line)
                elif line.lower().startswith(user_prefix.lower()) or line.lower().startswith('user:'):
                    current_mode = 'user'
                    # 移除user前缀
                    cleaned_line = re.sub(r'^(user|prompt):\s*', '', line, flags=re.IGNORECASE)
                    if cleaned_line:
                        user_parts.append(cleaned_line)
                else:
                    # 根据当前模式分类
                    if current_mode == 'system':
                        system_parts.append(line)
                    else:
                        user_parts.append(line)
        else:
            # 基于前缀的精确模式
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(system_prefix):
                    system_content = line[len(system_prefix):].strip()
                    if system_content:
                        system_parts.append(system_content)
                elif line.startswith(user_prefix):
                    user_content = line[len(user_prefix):].strip()
                    if user_content:
                        user_parts.append(user_content)
                else:
                    # 默认作为用户提示
                    user_parts.append(line)
        
        # 构建结果
        system_text = '\n'.join(system_parts).strip()
        user_text = '\n'.join(user_parts).strip()
        
        # Gemma提示：包含system和user（如果有system的话）
        if system_text and user_text:
            gemma_prompt = f"<system>\n{system_text}\n</system>\n\n{user_text}"
        elif system_text:
            gemma_prompt = f"<system>\n{system_text}\n</system>"
        else:
            gemma_prompt = user_text
        
        # Jina CLIP提示：仅用户内容（不需要system prompt）
        jina_prompt = user_text
        
        # 组合提示：原始文本
        combined_prompt = text
        
        return gemma_prompt, jina_prompt, combined_prompt
    
    def _is_system_prompt(self, line: str) -> bool:
        """
        判断是否为system prompt
        """
        line_lower = line.lower()
        
        # 常见的system prompt标识
        system_indicators = [
            'system:',
            'sys:',
            'instruction:',
            'instruct:',
            'you are',
            'act as',
            'behave as',
            'role:',
            'persona:',
            'character:',
        ]
        
        return any(line_lower.startswith(indicator) for indicator in system_indicators)


class NewBiePromptFormatter:
    """
    NewBie提示词格式化器
    为不同的编码器格式化提示词
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "System prompt for Gemma"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "User prompt"
                }),
            },
            "optional": {
                "gemma_template": (["chat", "instruct", "simple"], {
                    "default": "chat",
                    "description": "Template format for Gemma3-4B-IT"
                }),
                "add_generation_prompt": ("BOOLEAN", {
                    "default": True,
                    "description": "Add generation prompt for Gemma"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gemma_formatted", "jina_formatted")
    FUNCTION = "format_prompts"
    CATEGORY = "conditioning"
    TITLE = "NewBie Prompt Formatter"

    def format_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
        gemma_template: str = "chat",
        add_generation_prompt: bool = True
    ) -> Tuple[str, str]:
        """
        格式化提示词用于不同的编码器
        
        Returns:
            (gemma_formatted, jina_formatted)
        """
        
        # Jina CLIP格式：简单的用户提示
        jina_formatted = user_prompt.strip()
        
        # Gemma3-4B-IT格式
        if gemma_template == "chat":
            # 聊天格式
            if system_prompt.strip():
                gemma_formatted = f"<start_of_turn>system\n{system_prompt.strip()}<end_of_turn>\n"
            else:
                gemma_formatted = ""
            
            gemma_formatted += f"<start_of_turn>user\n{user_prompt.strip()}<end_of_turn>\n"
            
            if add_generation_prompt:
                gemma_formatted += "<start_of_turn>model\n"
                
        elif gemma_template == "instruct":
            # 指令格式
            if system_prompt.strip():
                gemma_formatted = f"### Instruction:\n{system_prompt.strip()}\n\n"
            else:
                gemma_formatted = ""
            
            gemma_formatted += f"### Input:\n{user_prompt.strip()}\n\n"
            
            if add_generation_prompt:
                gemma_formatted += "### Response:\n"
                
        else:  # simple
            # 简单格式
            if system_prompt.strip() and user_prompt.strip():
                gemma_formatted = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            elif system_prompt.strip():
                gemma_formatted = system_prompt.strip()
            else:
                gemma_formatted = user_prompt.strip()
        
        return gemma_formatted, jina_formatted


# 节点映射
NODE_CLASS_MAPPINGS = {
    "NewBiePromptSeparator": NewBiePromptSeparator,
    "NewBiePromptFormatter": NewBiePromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBiePromptSeparator": "NewBie Prompt Separator",
    "NewBiePromptFormatter": "NewBie Prompt Formatter",
}