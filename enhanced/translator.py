import os
import random
import re
import torch
import tarfile
import time
import translators as ts

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from modules.config import paths_llms
from modules.model_loader import load_file_from_url
from download import download
from functools import lru_cache
from modules.util import is_chinese
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

Q_punct = '｀～！＠＃＄％＾＆＊（）＿＋＝－｛｝［］：＂；｜＜＞？，．／。　１２３４５６７８９０'
B_punct = '`~!@#$%^&*()_+=-{}[]:";|<>?,./. 1234567890'
Q_alphabet = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
B_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

translator_org = ['alibaba','cloudTranslation', 'google', 'iflyrec', 'translateCom', 'youdao']
translator_default = 'alibaba'
translator_path = os.path.join(paths_llms[0], 'nllb-200-distilled-600M')
translator_slim_path = os.path.join(paths_llms[0], 'Helsinki-NLP/opus-mt-zh-en')

translator_path_old = os.path.join(paths_llms[0], '../translator')
if os.path.exists(translator_path_old) and not os.path.exists(paths_llms[0]):
    os.rename(translator_path_old, paths_llms[0])


g_tokenizer = ''
g_model = ''
g_model_type = ''

def singularize_word(word: str) -> str:
    if not word:
        return word

    lower = word.lower()
    oe_plural_s_only = {
        "shoes",
        "toes",
        "canoes",
        "oboes",
        "aloes",
        "floes",
    }
    irregular = {
        "men": "man",
        "women": "woman",
        "children": "child",
        "people": "person",
        "mice": "mouse",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
    }
    if lower in irregular:
        base = irregular[lower]
    elif len(lower) <= 3:
        base = lower
    elif lower.endswith("ies") and len(lower) > 4:
        base = lower[:-3] + "y"
    elif lower.endswith("oes") and lower in oe_plural_s_only:
        base = lower[:-1]
    elif lower.endswith(("ses", "xes", "zes", "ches", "shes", "oes")) and len(lower) > 4:
        base = lower[:-2]
    elif lower.endswith("s") and not lower.endswith(("ss", "us", "is", "as", "os")) and len(lower) > 3:
        base = lower[:-1]
    else:
        base = lower

    if word.isupper():
        return base.upper()
    if word[:1].isupper() and word[1:].islower():
        return base[:1].upper() + base[1:]
    return base

def normalize_prompt(prompt_text: str) -> str:
    if prompt_text is None:
        return prompt_text
    text = str(prompt_text).strip()
    if not text:
        return text

    parts = [p.strip() for p in re.split(r"[，,]", text) if p and p.strip()]
    normalized_parts = []
    for part in parts:
        words = part.split()
        normalized_words = []
        for w in words:
            m = re.match(r"^([^A-Za-z]*)([A-Za-z][A-Za-z'-]*)([^A-Za-z]*)$", w)
            if not m:
                normalized_words.append(w)
                continue
            prefix, core, suffix = m.group(1), m.group(2), m.group(3)
            normalized_words.append(prefix + singularize_word(core) + suffix)
        normalized_parts.append(" ".join(normalized_words).strip())

    return ", ".join([p for p in normalized_parts if p])

def Q2B_number_punctuation(text):
    global Q_punct, B_punct

    texts = list(text)
    Bpunct = list(B_punct)
    for i in range(0,len(texts)):
        j = Q_punct.find(texts[i])
        if j >= 0:
            texts[i] = Bpunct[j]
    return ''.join(texts)

def Q2B_alphabet(text):
    global Q_alphabet, B_alphabet

    texts = list(text)
    Balphabet = list(B_alphabet)
    for i in range(0,len(texts)):
        j = Q_alphabet.find(texts[i])
        if j >= 0:
            texts[i] = Balphabet[j]
    return ''.join(texts)


def translate2en_model(model, tokenizer, text_zh):
    inputs = tokenizer(text_zh, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=60
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()

def translate2zh_model(model, tokenizer, text_en):
    inputs = tokenizer(text_en, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"), max_length=60
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()


@lru_cache(maxsize=32, typed=False)
def translate2zh_apis(text):
    global translator_default
    if not text:
        return text
    try:
        return ts.translate_text(text, translator=translator_default, from_language='en', to_language='zh')
    except Exception as e:
        try:
            logger.info(f'Change another translator because of {e}')
            translator_default = translator_org[random.randint(0,5)]
            return ts.translate_text(text, translator=translator_default, from_language='en', to_language='zh')
        except Exception as e:
            logger.info(f'Error during translation of APIs methods: {e}')
            return text
def translate2en_apis(text):
    global translator_default
    if not text:
        return text
    try:
        return ts.translate_text(text, translator=translator_default, to_language='en')
    except Exception as e:
        try:
            logger.info(f'Change another translator because of {e}')
            translator_default = translator_org[random.randint(0,5)]
            return ts.translate_text(text, translator=translator_default, to_language='en')
        except Exception as e:
            logger.info(f'Error during translation of APIs methods: {e}')
            return text

def init_or_load_translator_model(method='Slim Model'):
    global g_tokenizer, g_model, g_model_type

    if 'g_tokenizer' not in globals():
        globals()['g_tokenizer'] = None
    if 'g_model' not in globals():
        globals()['g_model'] = None

    logger.info(f'init_or_load_translator_model: {method}')
    if method != g_model_type or g_tokenizer is None or g_model is None:
        if method == "Big Model":
            # NLLB-200 requires several files to work with AutoTokenizer and AutoModel
            required_files = [
                'config.json',
                'pytorch_model.bin',
                'tokenizer_config.json',
                'sentencepiece.bpe.model',
                'special_tokens_map.json',
                'tokenizer.json'
            ]
            hf_repo = "facebook/nllb-200-distilled-600M"

            if not os.path.exists(translator_path):
                os.makedirs(translator_path)
                url = 'https://gitee.com/metercai/SimpleSDXL/releases/download/win64/nllb_200_distilled_600m.tar.gz'
                cached_file = os.path.join(translator_path, 'nllb_200_distilled_600m.tar.gz')
                try:
                    download(url, cached_file, progressbar=True)
                    with tarfile.open(cached_file, 'r:gz') as tarf:
                        tarf.extractall(translator_path)
                    os.remove(cached_file)
                except Exception as e:
                    logger.warning(f"Failed to download or extract from Gitee: {e}. Will try Hugging Face.")

            # Check and download each missing file from Hugging Face
            for file_name in required_files:
                file_path = os.path.join(translator_path, file_name)
                if not os.path.exists(file_path):
                    load_file_from_url(
                        url=f'https://huggingface.co/{hf_repo}/resolve/main/{file_name}',
                        model_dir=translator_path,
                        file_name=file_name)

            logger.info(f'load model form : {translator_path}')
            g_tokenizer = AutoTokenizer.from_pretrained(translator_path, src_lang="zho_Hans")
            g_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path)
        else:
            # Opus-MT requires several files
            required_files = [
                'config.json',
                'pytorch_model.bin',
                'source.spm',
                'target.spm',
                'vocab.json',
                'tokenizer_config.json'
            ]
            hf_repo = "Helsinki-NLP/opus-mt-zh-en"

            if not os.path.exists(translator_slim_path):
                os.makedirs(translator_slim_path)
                url = 'https://gitee.com/metercai/SimpleSDXL/releases/download/win64/opus_mt_zh_en.tar.gz'
                cached_file = os.path.join(translator_slim_path, 'opus_mt_zh_en.tar.gz')
                try:
                    download(url, cached_file, progressbar=True)
                    with tarfile.open(cached_file, 'r:gz') as tarf:
                        tarf.extractall(translator_slim_path)
                    os.remove(cached_file)
                except Exception as e:
                    logger.warning(f"Failed to download or extract from Gitee: {e}. Will try Hugging Face.")

            # Check and download each missing file from Hugging Face
            for file_name in required_files:
                file_path = os.path.join(translator_slim_path, file_name)
                if not os.path.exists(file_path):
                    load_file_from_url(
                        url=f'https://huggingface.co/{hf_repo}/resolve/main/{file_name}',
                        model_dir=translator_slim_path,
                        file_name=file_name)

            logger.info(f'load slim model form : {translator_slim_path}')
            g_tokenizer = AutoTokenizer.from_pretrained(translator_slim_path)
            g_model = AutoModelForSeq2SeqLM.from_pretrained(translator_slim_path).eval()
        g_model_type = method
    return g_tokenizer, g_model

def free_translator_model():
    global g_tokenizer, g_model
    if 'g_tokenizer' in globals():
        del g_tokenizer
    if 'g_model' in globals():
        del g_model
    return

def toggle(text: str, method: str = 'Slim Model') -> str:
    is_chinese_ext = lambda x: (Q_alphabet + B_punct).find(x) < -1
    if is_chinese(text):
        return convert(text, method)
    else:
        return convert(text, method, 'cn')


def convert(text: str, method: str = 'Slim Model', lang: str = 'en' ) -> str:
    global Q_alphabet, B_puncti

    start = time.perf_counter()

    if lang=='cn':
        if method == 'Third APIs':
            text_zh = translate2zh_apis(text)
            ts_method = translator_default
        else:
            tokenizer, model = init_or_load_translator_model(method)
            text_zh = translate2zh_model(model, tokenizer, text)
            ts_method = method
        stop = time.perf_counter()
        logger.info(f'Translate by "{ts_method}" in {(stop-start):.2f}s: "{text}" to "{text_zh}"')
        return text_zh
    is_chinese_ext = lambda x: (Q_alphabet + B_punct).find(x) < -1 
    #text = Q2B_number_punctuation(text)
    if is_chinese(text):
        if method == 'Third APIs':
            logger.info(f'Using an online translation APIs.')
        else:
            tokenizer, model = init_or_load_translator_model(method)


        def T_ZH2EN(text_zh):
            if method=="Slim Model":
                encoded = tokenizer([text_zh], return_tensors="pt")
                sequences = model.generate(**encoded)
                return 'Slim Model', tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
            elif method=="Big Model":
                inputs = tokenizer(text_zh, return_tensors="pt")
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=60)
                return 'Big Model', tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].lower()
            else:
                return translator_default, translate2en_apis(text_zh)


        text_eng = ""
        text_zh = ""
        for _char in iter(text):
            if is_chinese(_char):
                text_zh += _char
            else:
                if len(text_zh) > 0:
                    if is_chinese_ext(_char):
                        text_zh += _char
                        continue
                    else:
                        #text_zh = Q2B_alphabet(text_zh)
                        ts_methods, text_en=T_ZH2EN(text_zh)
                        text_eng += text_en  
                        text_zh = ""
                text_eng += _char
        if len(text_zh) > 0:
            ts_methods, text_en=T_ZH2EN(text_zh)
            text_eng += text_en
        text_eng = Q2B_number_punctuation(text_eng)
        text_eng = Q2B_alphabet(text_eng)
        stop = time.perf_counter()
        logger.info(f'Translate by "{ts_methods}" in {(stop-start):.2f}s: "{text}" to "{text_eng}"')
        return text_eng
    return text


