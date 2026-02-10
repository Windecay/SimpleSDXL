from .network_translate.alibabav2 import AlibabaV2, AlibabaTranslateError
from .network_translate.bing import bing_translate
from .network_translate.wangyi import youdao_translate


def api_service_translate(text, from_lang='auto', to_lang='en', apiService='alibaba'):
    """
    统一翻译接口
    :param text: 待翻译文本（支持换行符分隔的批量翻译）
    :param from_lang: 源语言
    :param to_lang: 目标语言
    :param apiService: 选择翻译服务，支持 'alibaba', 'bing', 'youdao'
    :return: 翻译文本（批量翻译时返回换行符分隔的结果）
    """
    apiService = apiService.lower()
    if apiService == 'alibaba':
        translator = AlibabaV2()
        try:
            return translator.alibaba_api(text, from_language=from_lang, to_language=to_lang)
        except AlibabaTranslateError as e:
            return f"[Alibaba翻译失败] {e}"
    elif apiService == 'bing':
        try:
            result = bing_translate(text, from_lang=from_lang, to_lang=to_lang)
            return result[0]["translations"][0]["text"]
        except Exception as e:
            return f"[Bing翻译失败] {e}"
    elif apiService == 'youdao':
        try:
            if from_lang.lower() == 'zh':
                from_lang = 'zh-CHS'
            if to_lang.lower() == 'zh':
                to_lang = 'zh-CHS'
            _, translation = youdao_translate(text, from_lang, to_lang)
            return translation[0] if translation else ''
        except Exception as e:
            return f"[有道翻译失败] {e}"
    else:
        return f"不支持的翻译服务: {apiService}"


# 示例
if __name__ == "__main__":
    for service in ['alibaba', 'bing', 'youdao']:
        if service == 'youdao':
            translated = api_service_translate("你好，世界！", from_lang='zh-CHS', to_lang='en', apiService=service)
        else:
            translated = api_service_translate("你好，世界！", from_lang='zh', to_lang='en', apiService=service)
        print(f"{service} 翻译结果: {translated}")
