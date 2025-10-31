# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
# https://github.com/pythongosssss/ComfyUI-WD14-Tagger/blob/main/wd14tagger.py

# {
#     "wd-v1-4-moat-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2",
#     "wd-v1-4-convnextv2-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
#     "wd-v1-4-convnext-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2",
#     "wd-v1-4-convnext-tagger": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger",
#     "wd-v1-4-vit-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2"
# }


import numpy as np
import csv
import onnxruntime as ort
import os

from PIL import Image
from onnxruntime import InferenceSession
from modules.config import paths_clip_vision
from modules.model_loader import load_file_from_url
import logging
logger = logging.getLogger(__name__)

global_model = None
global_csv = None
current_model_name = None


def default_interrogator(image, threshold=0.35, character_threshold=0.85, exclude_tags=""):
    global global_model, global_csv, current_model_name

    new_model_name = "wd-eva02-large-tagger-v3"
    new_model_onnx_url = f'https://www.modelscope.cn/models/windecay/WD-tagger/resolve/master/{new_model_name}.onnx'
    new_model_csv_url = f'https://www.modelscope.cn/models/windecay/WD-tagger/resolve/master/{new_model_name}.csv'
    new_model_onnx_path = os.path.join(paths_clip_vision[0], f"{new_model_name}.onnx")
    new_model_csv_path = os.path.join(paths_clip_vision[0], f"{new_model_name}.csv")

    old_model_name = "wd-v1-4-moat-tagger-v2"
    old_model_onnx_url = f'https://huggingface.co/lllyasviel/misc/resolve/main/{old_model_name}.onnx'
    old_model_csv_url = f'https://huggingface.co/lllyasviel/misc/resolve/main/{old_model_name}.csv'

    use_new_model = os.path.exists(new_model_onnx_path) and os.path.exists(new_model_csv_path)

    if current_model_name != (new_model_name if use_new_model else old_model_name):
        global_model = None
        global_csv = None

    if use_new_model:
        model_name = new_model_name
        model_onnx_url = new_model_onnx_url
        model_csv_url = new_model_csv_url
        current_model_name = new_model_name
        logger.info(f"[WD14 Tagger] 当前使用模型: {model_name}")
    else:
        model_name = old_model_name
        model_onnx_url = old_model_onnx_url
        model_csv_url = old_model_csv_url
        current_model_name = old_model_name
        logger.info(f"[WD14 Tagger] 当前使用旧版模型: {model_name}，可运行模型检测更新。")
    model_onnx_filename = load_file_from_url(
        url=model_onnx_url,
        model_dir=paths_clip_vision[0],
        file_name=f'{model_name}.onnx',
    )

    model_csv_filename = load_file_from_url(
        url=model_csv_url,
        model_dir=paths_clip_vision[0],
        file_name=f'{model_name}.csv',
    )

    if global_model is not None:
        model = global_model
    else:
        model = InferenceSession(model_onnx_filename, providers=ort.get_available_providers())
        global_model = model

    input = model.get_inputs()[0]
    height = input.shape[1]
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    if global_csv is not None:
        csv_lines = global_csv
    else:
        csv_lines = []
        with open(model_csv_filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_lines.append(row)
        global_csv = csv_lines

    tags = []
    general_index = None
    character_index = None
    for line_num, row in enumerate(csv_lines):
        if general_index is None and row[2] == "0":
            general_index = line_num
        elif character_index is None and row[2] == "4":
            character_index = line_num
        tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all)).replace('_', ' ')
    return res
