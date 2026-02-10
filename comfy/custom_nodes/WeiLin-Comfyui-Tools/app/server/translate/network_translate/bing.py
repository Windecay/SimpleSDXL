# Bing翻译
import requests
import time
import threading

class BingTokenManager:
    _token = None
    _token_time = 0
    _lock = threading.Lock()
    _token_ttl = 3 * 60 * 60  # 3小时

    @classmethod
    def get_token(cls):
        with cls._lock:
            now = time.time()
            if cls._token is None or now - cls._token_time > cls._token_ttl:
                cls._token = cls._fetch_token()
                cls._token_time = now
            return cls._token

    @staticmethod
    def _fetch_token():
        url = "https://edge.microsoft.com/translate/auth"
        headers = {
            "accept": "*/*",
            "accept-language": "zh-TW,zh;q=0.9,ja;q=0.8,zh-CN;q=0.7,en-US;q=0.6,en;q=0.5",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referrer-policy": "strict-origin-when-cross-origin",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "none",
            "sec-fetch-storage-access": "active",
            "sec-mesh-client-arch": "x86_64",
            "sec-mesh-client-edge-channel": "stable",
            "sec-mesh-client-edge-version": "143.0.3650.80",
            "sec-mesh-client-os": "Windows",
            "sec-mesh-client-os-version": "10.0.26300",
            "sec-mesh-client-webview": "0",
            "x-edge-shopping-flag": "1",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text.strip()

def bing_translate(text, from_lang="zh-Hans", to_lang="en"):
    token = BingTokenManager.get_token()
    url = f"https://api-edge.cognitive.microsofttranslator.com/translate?from={from_lang}&to={to_lang}&api-version=3.0&includeSentenceLength=true"
    headers = {
        "accept": "*/*",
        "accept-language": "zh-TW,zh;q=0.9,ja;q=0.8,zh-CN;q=0.7,en-US;q=0.6,en;q=0.5",
        "authorization": f"Bearer {token}",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referrer-policy": "strict-origin-when-cross-origin",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "none",
        "sec-fetch-storage-access": "active"
    }
    body = [{"Text": text}]
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    result = bing_translate("你好，我的世界！")
    # 提取翻译文本
    try:
        text = result[0]["translations"][0]["text"]
        print("翻译文本:", text)
    except Exception as e:
        print("解析翻译结果出错:", e)