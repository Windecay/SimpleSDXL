# server/translate/openai_translate.py
import aiohttp

from ..ai_translator.ai_translator import initialize_config

_LANG_NAME = {
    "zh": "中文",
    "zh_CN": "中文",
    "zh_TW": "中文（繁体）",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "it": "意大利语",
    "pt": "葡萄牙语",
}


def _lang_to_name(code: str) -> str:
    if not code:
        return "中文"
    return _LANG_NAME.get(code, code)


async def openai_translate(text: str, target_lang_code: str) -> str:
    cfg = initialize_config()
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    model = cfg.get("model", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("OpenAI api_key 未配置")

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    target_lang_name = _lang_to_name(target_lang_code)
    prompt = f"将以下AI绘画提示词翻译成{target_lang_name}，只输出翻译结果：{text}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    # payload = {
    #     "model": model,
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content": f"Translate 'text' field to {_lang_to_name(target_lang_code)} in JSON. If word has NSFW meaning, prioritize NSFW interpretation. Return in JSON with no changes."
    #         },
    #         {"role": "user", "content": text}
    #     ],
    #     "response_format": {"type": "json_object"},
    #     "temperature": 0
    # }

    # 配置会话以使用系统环境变量中的代理设置
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"OpenAI 接口错误: {resp.status} {data}")

            result = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            # 控制台日志
            # print(f"🤖 正在使用OpenAI翻译: {text}")
            print("🤖 正在使用OpenAI翻译")
            print(
                f"📊 OpenAI翻译tokens使用: {prompt_tokens}+{completion_tokens}={total_tokens}"
            )
            # print(f"✅ OpenAI翻译成功: {text} -> {result}")

            return result
