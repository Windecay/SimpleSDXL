import json
import sys
import os
import io
import base64
import datetime
import hashlib
import hmac
import requests
import time
from PIL import Image
import torch
from .imagefunc import tensor2pil, pil2tensor, log, get_api_key, fit_resize_image

jimeng_i2i_model_list = ["jimeng_i2i_v30"]
jimeng_t2i_model_list = ["jimeng_high_aes_general_v21_L"]


method = 'POST'
host = 'visual.volcengineapi.com'
region = 'cn-north-1'
endpoint = 'https://visual.volcengineapi.com'
service = 'cv'
image_max_side_length = 4096

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, dateStamp, regionName, serviceName):
    kDate = sign(key.encode('utf-8'), dateStamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'request')
    return kSigning

def formatQuery(parameters):
    request_parameters_init = ''
    for key in sorted(parameters):
        request_parameters_init += key + '=' + parameters[key] + '&'
    request_parameters = request_parameters_init[:-1]
    return request_parameters


def signV4Request(access_key, secret_key, service, req_query, req_body):
    if access_key is None or secret_key is None:
        print('No access key is available.')
        sys.exit()

    t = datetime.datetime.utcnow()
    current_date = t.strftime('%Y%m%dT%H%M%SZ')
    # current_date = '20210818T095729Z'
    datestamp = t.strftime('%Y%m%d')  # Date w/o time, used in credential scope
    canonical_uri = '/'
    canonical_querystring = req_query
    signed_headers = 'content-type;host;x-content-sha256;x-date'
    payload_hash = hashlib.sha256(req_body.encode('utf-8')).hexdigest()
    content_type = 'application/json'
    canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + host + \
                        '\n' + 'x-content-sha256:' + payload_hash + \
                        '\n' + 'x-date:' + current_date + '\n'
    canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + \
                        '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
    # print(canonical_request)
    algorithm = 'HMAC-SHA256'
    credential_scope = datestamp + '/' + region + '/' + service + '/' + 'request'
    string_to_sign = algorithm + '\n' + current_date + '\n' + credential_scope + '\n' + hashlib.sha256(
        canonical_request.encode('utf-8')).hexdigest()
    # print(string_to_sign)
    signing_key = getSignatureKey(secret_key, datestamp, region, service)
    # print(signing_key)
    signature = hmac.new(signing_key, (string_to_sign).encode(
        'utf-8'), hashlib.sha256).hexdigest()
    # print(signature)

    authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + \
                           credential_scope + ', ' + 'SignedHeaders=' + \
                           signed_headers + ', ' + 'Signature=' + signature
    # print(authorization_header)
    headers = {'X-Date': current_date,
               'Authorization': authorization_header,
               'X-Content-Sha256': payload_hash,
               'Content-Type': content_type
               }
    # print(headers)

    # ************* SEND THE REQUEST *************
    request_url = endpoint + '?' + canonical_querystring

    # print('\nBEGIN REQUEST++++++++++++++++++++++++++++++++++++')
    # print('Request URL = ' + request_url)
    try:
        r = requests.post(request_url, headers=headers, data=req_body)
    except Exception as err:
        print(f'error occurred: {err}')
        raise
    else:
        # print('\nRESPONSE++++++++++++++++++++++++++++++++++++')
        # print(f'Response code: {r.status_code}\n')
        # 使用 replace 方法将 \u0026 替换为 &
        resp_str = r.text.replace("\\u0026", "&")
        # print(f'Response body: {resp_str}\n')
        return json.loads(resp_str)

def round_to_multiple_of_16(x):
    return int(round(x / 16)) * 16

def get_closest_api_size(orig_width, orig_height):
    """
    根据原始图片尺寸，返回最接近的API尺寸。
    """
    aspect = orig_width / orig_height
    max_width, max_height = 2016, 1536
    max_aspect = max_width / max_height

    if aspect > max_aspect:
        # 限制宽度
        width = max_width
        height = width / aspect
    else:
        # 限制高度
        height = max_height
        width = aspect * height

    # 四舍五入到16的倍数
    width = round_to_multiple_of_16(width)
    height = round_to_multiple_of_16(height)

    # 再次确保合法范围
    width = min(max(width, 512), 2016)
    height = min(max(height, 512), 1536)

    return width, height



class LS_Jimeng_i2i_API:

    def __init__(self):
        self.NODE_NAME = 'Jimeng Image2Image API'
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (jimeng_i2i_model_list,),
                "time_out": ("INT", {"default": 300, "min": 1, "max": 3600, "step": 1}), # 300s = 5min
                "scale": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'run_jimeng_i2i_api'
    CATEGORY = '😺dzNodes/LayerUtility'

    def run_jimeng_i2i_api(self, image, model, time_out, scale, seed, prompt):
        ret_images = []
        access_key = get_api_key("volcengine_AccessKeyId")
        secret_key = get_api_key("volcengine_SecretAccessKey")

        # image shape = b, h, w, c
        orig_width = image.shape[2]
        orig_height = image.shape[1]
        output_width, output_height = get_closest_api_size(orig_width, orig_height)

        for img in image:
            img = torch.unsqueeze(img, 0)
            orig_image = tensor2pil(img).convert('RGB')
            orig_image = fit_resize_image(orig_image, output_width, output_height, 'fill', Image.BICUBIC)
            buffered = io.BytesIO()
            orig_image.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            body_params = {
                "req_key": model,
                "binary_data_base64": [image_base64],
                "prompt": prompt,
                "seed": seed,
                "scale": scale,
                "width": output_width,
                "height": output_height,
            }
            formatted_body = json.dumps(body_params)

            query_params = {
                'Action': 'CVSync2AsyncSubmitTask',
                'Version': '2022-08-31',
            }
            formatted_query = formatQuery(query_params)

            # 发送请求

            result = signV4Request(access_key, secret_key, service, formatQuery(query_params), formatted_body)
            response_code = result["code"]
            if response_code > 10000:
                print(f"Error: {result}")
                raise Exception(f"error code: {response_code}")
            task_id = result["data"]["task_id"]
            log(f"Send Request to Jimeng API: task_id = {task_id}")

            # 获取结果
            start_time = time.time()
            check_interval = 1
            check_timeout = 60

            query_params["Action"] = 'CVSync2AsyncGetResult'
            formatted_query = formatQuery(query_params)

            body_params = {
                "req_key": "jimeng_i2i_v30",
                "task_id": task_id,
            }
            formatted_body = json.dumps(body_params)

            while True:
                time.sleep(check_interval)
                result = signV4Request(access_key, secret_key, service, formatted_query, formatted_body)
                query_status = result["data"]["status"]
                if query_status == "done":
                    # print(f"query_status: {query_status}")
                    break
                if time.time() - start_time > check_timeout:
                    raise Exception("Jimeng API Timeout")

            end_time = time.time()
            use_time = round(end_time - start_time, 2)
            log(f"Jimeng API responded in {use_time} seconds.")

            base64_str = result["data"]["binary_data_base64"][0]
            image_data = base64.b64decode(base64_str)
            ret_image = Image.open(io.BytesIO(image_data))
            ret_images.append(pil2tensor(ret_image))

        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: JimengI2IAPI": LS_Jimeng_i2i_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: JimengI2IAPI": "LayerUtility: Jimeng Imgae to Image API (Advance)",
}