import os
import re
import json
import math
import gradio as gr
import enhanced.translator as translator
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))

from modules.util import get_files_from_folder
from args_manager import args

wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../wildcards/'))
wildcards_max_bfs_depth = 64

wildcards = {}
wildcards_list = {}
wildcards_translation = {}
wildcards_words_translation = {}
wildcards_template = {}
wildcards_weight_range = {}
wildcards_cache = {}

array_regex = re.compile(r'\[([\w\(\)\.\s,;:-]+)\]')
array_regex1 = re.compile(r'\[([\w\(\)\s\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF,;.:\"\'-]+)\]')
tag_regex0 = re.compile(r'([\s\w\(\);-]+)')
tag_regex1 = re.compile(r'([\s\w\(\),-]+)')
tag_regex2 = re.compile(r'__([\w-]+)__')
tag_regex3 = re.compile(r'__([\w-]+)__:([\d]+)')
tag_regex4 = re.compile(r'__([\w-]+)__:([RLrl]){1}([\d]*)')
tag_regex5 = re.compile(r'__([\w-]+)__:([RLrl]){1}([\d]*):([\d]+)')
tag_regex6 = re.compile(r'__([\w-]+)__:([\d]+):([\d]+)')

wildcard_regex = re.compile(r'-([\w-]+)-')

def set_wildcard_path_list(name, list_value):
    global wildcards_list
    if name in wildcards_list.keys():
        if list_value not in wildcards_list[name]:
            wildcards_list[name].append(list_value)
    else:
        wildcards_list.update({name: [list_value]})

def _get_cache_key(user_did):
    return user_did if user_did else "__public__"

def _is_guest_user(user_did):
    try:
        import shared
        if shared.token is None or user_did is None:
            return True
        return shared.token.is_guest(user_did)
    except Exception:
        return True

def _get_user_wildcards_dir(user_did):
    if user_did is None or _is_guest_user(user_did):
        return None
    try:
        import shared
        if shared.token is None:
            return None
        path = shared.token.get_path_in_user_dir(user_did, "wildcards")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        return None

def _get_public_wildcards_dirs():
    dirs = []
    if os.path.isdir(wildcards_path):
        dirs.append(wildcards_path)
    try:
        import modules.config as config
        for p in getattr(config, "paths_wildcards", []) or []:
            ap = os.path.abspath(p)
            if os.path.isdir(ap) and ap not in dirs:
                dirs.append(ap)
    except Exception:
        pass
    return dirs

def _get_wildcards_dirs(user_did=None):
    dirs = []
    user_dir = _get_user_wildcards_dir(user_did)
    if user_dir and os.path.isdir(user_dir):
        dirs.append(user_dir)
    dirs += _get_public_wildcards_dirs()
    return dirs

def _to_wildcard_key(rel_path):
    if rel_path.lower().endswith(".txt"):
        rel_path = rel_path[:-4]
    return rel_path.replace("\\", "/")

def _build_wildcards_context(user_did=None):
    dirs = _get_wildcards_dirs(user_did)
    wildcard_sources = {}
    signature_items = []

    for root_dir in dirs:
        try:
            files = get_files_from_folder(root_dir, ['.txt'], None, variation=False)
        except Exception:
            continue
        for rel in files:
            key = _to_wildcard_key(rel)
            if key in wildcard_sources:
                continue
            full_path = os.path.join(root_dir, rel)
            if not os.path.isfile(full_path):
                continue
            wildcard_sources[key] = full_path
            try:
                signature_items.append((root_dir, rel.replace("\\", "/"), int(os.path.getmtime(full_path))))
            except Exception:
                signature_items.append((root_dir, rel.replace("\\", "/"), 0))

    signature = tuple(sorted(signature_items))

    ctx = {
        "user_did": user_did,
        "dirs": dirs,
        "signature": signature,
        "wildcards": {},
        "wildcards_list": {},
        "wildcards_template": {},
        "wildcards_weight_range": {},
    }

    for wildcard, file_path in sorted(wildcard_sources.items(), key=lambda x: x[0].casefold()):
        try:
            words = open(file_path, encoding='utf-8').read().splitlines()
        except Exception:
            words = []
        words = [x.split('?')[0] for x in words if x != '' and not wildcard_regex.findall(x)]

        templates = [x for x in words if '|' in x]
        for line in templates:
            parts = line.split("|")
            word = parts[0]
            template = parts[1] if len(parts) > 1 else ''
            weight_range = parts[2] if len(parts) > 2 else ''
            if word is None or word == '':
                ctx["wildcards_template"][wildcard] = template
                if len(weight_range.strip()) > 0:
                    ctx["wildcards_weight_range"][wildcard] = weight_range
            else:
                ctx["wildcards_template"][f'{wildcard}/{word}'] = template
                if len(weight_range.strip()) > 0:
                    ctx["wildcards_weight_range"][f'{wildcard}/{word}'] = weight_range

        words = [x.split("|")[0] for x in words]
        ctx["wildcards"][wildcard] = words

    wildcards_list_local = {}
    def set_list(name, list_value):
        if name in wildcards_list_local:
            if list_value not in wildcards_list_local[name]:
                wildcards_list_local[name].append(list_value)
        else:
            wildcards_list_local[name] = [list_value]

    for wildcard in ctx["wildcards"].keys():
        wildcard_path = wildcard.split("/")
        if len(wildcard_path) == 1:
            set_list("root", wildcard_path[0])
        elif len(wildcard_path) == 2:
            set_list(wildcard_path[0], wildcard_path[1])
        elif len(wildcard_path) == 3:
            set_list(f'{wildcard_path[0]}/{wildcard_path[1]}', wildcard_path[2])
            set_list(wildcard_path[0], wildcard_path[1])
        else:
            logger.info(f'The level of wildcards is too depth: {wildcard}.')

    for k in list(wildcards_list_local.keys()):
        wildcards_list_local[k] = sorted(wildcards_list_local[k], key=lambda s: s.casefold())

    ctx["wildcards_list"] = wildcards_list_local
    return ctx

def ensure_wildcards_loaded(user_did=None, reload_flag=False):
    global wildcards_cache, wildcards, wildcards_list, wildcards_template, wildcards_weight_range
    cache_key = _get_cache_key(user_did)
    ctx = wildcards_cache.get(cache_key, None)
    if ctx is None or reload_flag:
        new_ctx = _build_wildcards_context(user_did)
        wildcards_cache[cache_key] = new_ctx
        ctx = new_ctx
    else:
        new_ctx = _build_wildcards_context(user_did)
        if new_ctx["signature"] != ctx.get("signature"):
            wildcards_cache[cache_key] = new_ctx
            ctx = new_ctx

    if user_did is None:
        wildcards = ctx["wildcards"]
        wildcards_list = ctx["wildcards_list"]
        wildcards_template = ctx["wildcards_template"]
        wildcards_weight_range = ctx["wildcards_weight_range"]
    return ctx

def get_wildcards_samples(path="root", trans=True, user_did=None):
    global wildcards_path, wildcards_translation

    ctx = ensure_wildcards_loaded(user_did)
    if path not in ctx["wildcards_list"] or len(ctx["wildcards_list"][path]) == 0:
        return []

    if not trans:
        return [[x] for x in ctx["wildcards_list"][path]]

    if args.language == 'cn':
        if len(wildcards_translation.keys()) == 0:
            wildcards_translation_file = os.path.join(wildcards_path, 'cn_list.json')
            if os.path.exists(wildcards_translation_file):
                with open(wildcards_translation_file, "r", encoding="utf-8") as json_file:
                    wildcards_translation.update(json.load(json_file))

    return [[get_wildcard_translation(x)] for x in ctx["wildcards_list"][path]]

get_wildcard_translation = lambda x: x if args.language!='cn' or f'list/{x}' not in wildcards_translation else wildcards_translation[f'list/{x}']

def load_words_translation(reload_flag=False):
    global wildcards_path, wildcards_words_translation
    if len(wildcards_words_translation.keys())==0 or reload_flag:
        translation_file = os.path.join(wildcards_path, 'cn_words.json')
        if os.path.exists(translation_file):
            with open(translation_file, "r", encoding="utf-8") as json_file:
                wildcards_words_translation.update(json.load(json_file))

def get_words_of_wildcard_samples(wildcard="root", user_did=None):
    global wildcards_words_translation

    ctx = ensure_wildcards_loaded(user_did)
    if wildcard == "root":
        root_list = ctx["wildcards_list"].get("root", [])
        if len(root_list) == 0:
            return []
        wildcard = root_list[0]

    words_source = ctx["wildcards"].get(wildcard, [])
    if args.language == 'cn':
        if len(wildcards_words_translation.keys()) == 0:
            load_words_translation()
        return [[x if x not in wildcards_words_translation else wildcards_words_translation[x]] for x in words_source]
    return [[x] for x in words_source]

def get_words_with_wildcard(wildcard, rng, method='R', number=1, start_at=1, user_did=None):
    ctx = ensure_wildcards_loaded(user_did)

    if wildcard is None or wildcard=='':
        words = []
    else:
        words = ctx["wildcards"].get(wildcard, [])
    words_result = []
    number0 = number
    if method=='L' or method=='l':
        if number == 0:
            words_result = words
        else:
            if number < 0:
                number = 1
            start = start_at - 1
            if number > len(words):
                number = len(words)
            if (start + number)>len(words):
                words_result = words[start:] + words[:start + number - len(words)]
            else:
                words_result = words[start:start + number]
    else:
        if number < 1:
            number = 1
        if number > len(words):
            number = len(words)
        nums = 1 if start_at<=1 else start_at
        for i in range(number):
            words_each = rng.sample(words, nums)
            words_result.append(words_each[0] if nums==1 else ", ".join(words_each))
    words_result = [replace_wildcard(txt, rng, user_did=user_did) for txt in words_result]
    logger.info(f'Get words from wildcard:__{wildcard}__, method:{method}, number:{number}, start_at:{start_at}, result:{words_result}')
    return words_result


def compile_arrays(text, rng, user_did=None):
    global wildcards, wildcards_max_bfs_depth, array_regex, array_regex1, tag_regex1, tag_regex2, tag_regex3, tag_regex4, tag_regex5, tag_regex6

    _ = ensure_wildcards_loaded(user_did)
    tag_arrays = array_regex1.findall(text)
    arrays = []
    mult = 1
    seed_fixed = True
    has_active_arrays = False

    if len(tag_arrays) > 0:
        for tag in tag_arrays:
            tag = tag.strip()
            if tag == '':
                arrays.append([''])
                continue

            if '__' in tag:
                has_active_arrays = True
                colon_counter = tag.count(':')
                wildcard = ''
                number = 1
                method = 'R'
                start_at = 1
                found = False

                if colon_counter >= 2:
                    parts = tag_regex5.findall(tag)
                    if parts:
                        parts = list(parts[0])
                        wildcard = parts[0]
                        method = parts[1]
                        if parts[2]:
                            number = int(parts[2])
                        start_at = int(parts[3])
                        found = True
                    else:
                        parts = tag_regex6.findall(tag)
                        if parts:
                            parts = list(parts[0])
                            wildcard = parts[0]
                            number = int(parts[1])
                            start_at = int(parts[2])
                            found = True

                if not found and colon_counter >= 1:
                    parts = tag_regex4.findall(tag)
                    if parts:
                        parts = list(parts[0])
                        wildcard = parts[0]
                        method = parts[1]
                        if parts[2]:
                            number = int(parts[2])
                        found = True
                    else:
                        parts = tag_regex3.findall(tag)
                        if parts:
                            parts = list(parts[0])
                            wildcard = parts[0]
                            number = int(parts[1])
                            found = True

                if not found:
                    sub_parts = tag_regex2.findall(tag)
                    if sub_parts:
                        wildcard = sub_parts[0]
                        found = True

                if found and wildcard:
                    words = get_words_with_wildcard(wildcard, rng, method, number, start_at, user_did=user_did)
                    arrays.append(words if len(words) > 0 else [''])
                    if not method.isupper():
                        seed_fixed = False
                else:
                    delimiter = ';' if ';' in tag else ','
                    words = [x.strip() for x in tag.split(delimiter) if x.strip() != '']
                    arrays.append(words if len(words) > 0 else [''])
                    if delimiter == ';':
                        seed_fixed = False
            else:
                if ',' in tag or ';' in tag:
                    has_active_arrays = True
                    delimiter = ';' if ';' in tag else ','
                    words = [x.strip() for x in tag.split(delimiter) if x.strip() != '']
                    arrays.append(words if len(words) > 0 else [''])
                    if delimiter == ';':
                        seed_fixed = False
                else:
                    arrays.append([f'[{tag}]'])

        for arr in arrays:
            mult *= max(1, len(arr))

    if (len(arrays) == 0) or (not has_active_arrays):
        arrays = []
        mult = 0

    # Support for naked wildcards with parameters (e.g. __wildcard__:3)
    def get_replacement(wildcard, method, number, start_at):
        words = get_words_with_wildcard(wildcard, rng, method, number, start_at, user_did=user_did)
        delimiter = ',' if method.isupper() else ';'
        if delimiter == ';':
            nonlocal seed_fixed
            seed_fixed = False
        joiner = ', ' if delimiter == ',' else '; '
        return joiner.join(words)

    def sub_outside_arrays(pattern, repl, input_text):
        parts = []
        last = 0
        for m in array_regex1.finditer(input_text):
            outside = input_text[last:m.start()]
            parts.append(pattern.sub(repl, outside))
            parts.append(input_text[m.start():m.end()])
            last = m.end()
        parts.append(pattern.sub(repl, input_text[last:]))
        return ''.join(parts)

    # Regex 5: __name__:M[N]:S
    text = sub_outside_arrays(tag_regex5, lambda m: get_replacement(
        m.group(1), m.group(2), int(m.group(3)) if m.group(3) else 1, int(m.group(4))
    ), text)

    # Regex 6: __name__:N:S
    text = sub_outside_arrays(tag_regex6, lambda m: get_replacement(
        m.group(1), 'R', int(m.group(2)), int(m.group(3))
    ), text)

    # Regex 4: __name__:M[N]
    text = sub_outside_arrays(tag_regex4, lambda m: get_replacement(
        m.group(1), m.group(2), int(m.group(3)) if m.group(3) else 1, 1
    ), text)

    # Regex 3: __name__:N
    text = sub_outside_arrays(tag_regex3, lambda m: get_replacement(
        m.group(1), 'R', int(m.group(2)), 1
    ), text)

    logger.info(f'Copmile text in prompt to arrays: {text} -> arrays:{arrays}, mult:{mult}')
    return text, arrays, mult, seed_fixed

def replace_wildcard(text, rng, user_did=None):
    global wildcards_max_bfs_depth, tag_regex2
    ctx = ensure_wildcards_loaded(user_did)
    parts = tag_regex2.findall(text)
    i = 1
    while parts:
        for wildcard in parts:
            if wildcard in ctx["wildcards"]:
                text = text.replace(f'__{wildcard}__', rng.choice(ctx["wildcards"][wildcard]), 1)
        parts = tag_regex2.findall(text)
        i += 1
        if i > wildcards_max_bfs_depth:
            break
    return text


def get_words(arrays, totalMult, index):
    if(len(arrays) == 1):
        word = arrays[0][index]
        #if word[0] == '(' and word[-1] == ')':
        #    word = word[1:-1]
        return [word]
    else:
        words = arrays[0]
        word = words[index % len(words)]
        #if word[0] == '(' and word[-1] == ')':
        #    word = word[1:-1]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(totalMult/len(words)), index)


def apply_arrays(text, index, arrays, mult):
    if len(arrays) == 0 or mult == 0:
        return text
    
    tags = array_regex1.findall(text)

    index %= mult
    chosen_words = get_words(arrays, mult, index)

    i = 0
    for arr in arrays:
        if i<len(tags) and i<len(chosen_words):
            if not tag_regex2.findall(chosen_words[i]):
                text = text.replace(f'[{tags[i]}]', chosen_words[i], 1)
            else:
                text = text.replace(f'[{tags[i]}]', tags[i], 1)
        i = i+1

    return text


def apply_wildcards(wildcard_text, rng, user_did=None):
    global tag_regex2
    ctx = ensure_wildcards_loaded(user_did)

    for _ in range(wildcards_max_bfs_depth):
        placeholders = tag_regex2.findall(wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        logger.info(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                words = ctx["wildcards"][placeholder]
                assert len(words) > 0
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                logger.info(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            logger.info(f'[Wildcards] {wildcard_text}')

    logger.info(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text


def _get_user_did_from_state(state_params):
    try:
        if isinstance(state_params, dict) and "user" in state_params and state_params["user"] is not None:
            user = state_params["user"]
            if hasattr(user, "get_did"):
                return user.get_did()
    except Exception:
        pass
    return None

def add_wildcards_and_array_to_prompt(wildcard, prompt, state_params):
    user_did = _get_user_did_from_state(state_params)
    ctx = ensure_wildcards_loaded(user_did)
    root_list = ctx["wildcards_list"].get("root", [])
    if not root_list:
        return gr.update(value=prompt), gr.Dataset.update(label=':', samples=[]), gr.update(open=True)

    wildcard = root_list[wildcard]
    state_params.update({"wildcard_in_wildcards": wildcard})
    if len(prompt)>0:
        if prompt[-1]=='[':
            state_params["array_wildcards_mode"] = '['
            prompt = prompt[:-1]
        elif prompt[-1]=='_':
            state_params["array_wildcards_mode"] = '_'
            if len(prompt)==1 or len(prompt)>2 and prompt[-2]!='_':
                prompt = prompt[:-1]
        else:
            state_params["array_wildcards_mode"] = '_'
    else:
        state_params["array_wildcards_mode"] = '_'
    
    if state_params["array_wildcards_mode"] == '[':
        new_tag = f'[__{wildcard}__]'
    else:
        new_tag = f'__{wildcard}__'
    prompt = f'{prompt.strip()} {new_tag}'
    return gr.update(value=prompt), gr.Dataset.update(label=f'{get_wildcard_translation(wildcard)}:', samples=get_words_of_wildcard_samples(wildcard, user_did=user_did)), gr.update(open=True)

def add_word_to_prompt(wildcard, index, prompt, state_params):
    user_did = _get_user_did_from_state(state_params)
    ctx = ensure_wildcards_loaded(user_did)
    root_list = ctx["wildcards_list"].get("root", [])
    if not root_list:
        return gr.update(value=prompt)

    wildcard = root_list[wildcard]
    words = ctx["wildcards"].get(wildcard, [])
    if index < 0 or index >= len(words):
        return gr.update(value=prompt)
    word = words[index]
    prompt = prompt.strip()
    for tag in [f'[__{wildcard}__]', f'__{wildcard}__']:
        if prompt.endswith(tag):
            prompt = prompt[:-1*len(tag)]
            break
    prompt = f'{prompt.strip()} {word}'
    return gr.update(value=prompt)

def normalize_int(v, default_value=1, min_value=1):
    try:
        iv = int(v)
    except Exception:
        return default_value
    return max(min_value, iv)

def build_wildcards_helper_tag(target, method, seed_mode, name, count, start, group_size):
    name = "" if name is None else str(name).strip()
    if name == "":
        return ""

    count = normalize_int(count, 1, 1)
    start = normalize_int(start, 1, 1)
    group_size = normalize_int(group_size, 1, 1)

    fixed_seed = (seed_mode == "Fixed seed")
    in_order = (method == "In order")
    method_letter = ("L" if fixed_seed else "l") if in_order else ("R" if fixed_seed else "r")

    if target == "Single in prompt":
        if in_order:
            return f"__{name}__:{method_letter}{count}:{start}"
        if group_size > 1:
            return f"__{name}__:{method_letter}{count}:{group_size}"
        if count > 1:
            return f"__{name}__:{method_letter}{count}"
        return f"__{name}__"

    if in_order:
        return f"[__{name}__:{method_letter}{count}:{start}]"
    if group_size > 1:
        return f"[__{name}__:{method_letter}{count}:{group_size}]"
    return f"[__{name}__:{method_letter}{count}]"

def update_wildcards_helper_controls(target, method, *_):
    show_start = (method == "In order")
    return gr.update(visible=show_start), gr.update(visible=(not show_start))

def update_wildcards_helper_preview(target, method, seed_mode, name, count, start, group_size):
    tag = build_wildcards_helper_tag(target, method, seed_mode, name, count, start, group_size)
    if tag == "":
        return ""
    return f"<div><b>Preview</b>: <code>{tag}</code></div>"

def append_wildcards_helper_tag_to_prompt(prompt_text, target, method, seed_mode, name, count, start, group_size):
    tag = build_wildcards_helper_tag(target, method, seed_mode, name, count, start, group_size)
    if tag == "":
        return prompt_text
    prompt_text = "" if prompt_text is None else str(prompt_text)
    if prompt_text.strip() == "":
        return tag
    return f"{prompt_text.strip()} {tag}"

def refresh_wildcards_components(state_params):
    user_did = _get_user_did_from_state(state_params)
    samples = get_wildcards_samples(user_did=user_did)
    names = [x[0] for x in get_wildcards_samples(trans=False, user_did=user_did)]
    words = get_words_of_wildcard_samples("root", user_did=user_did)
    name_value = names[0] if len(names) > 0 else ""
    return (
        gr.Dataset.update(samples=samples),
        gr.update(choices=names, value=name_value),
        gr.Dataset.update(samples=words),
    )

