# 本工具来自 https://github.com/UlionTse/translators/blob/master/translators/server.py
# 又WeiLin二次修改以适配本项目使用

import re
import time
import sys
import urllib.parse
import warnings
import functools
from typing import Optional, Tuple, Union
import requests

class AlibabaTranslateError(Exception):
    """自定义异常：阿里巴巴翻译相关错误"""
    pass
LangMapKwargsType = Union[str, bool]
ApiKwargsType = Union[str, int, float, bool, dict]
SessionType = Union[requests.sessions.Session]
ResponseType = Union[requests.models.Response]

class AlibabaV2:

    @staticmethod
    def time_stat(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if_show_time_stat = kwargs.get('if_show_time_stat', False)
            show_time_stat_precision = kwargs.get(
                'show_time_stat_precision', 2)
            sleep_seconds = kwargs.get('sleep_seconds', 0)

            if if_show_time_stat and sleep_seconds >= 0:
                t1 = time.time()
                result = func(*args, **kwargs)
                t2 = time.time()
                cost_time = round((t2 - t1 - sleep_seconds),
                                  show_time_stat_precision)
                sys.stderr.write(
                    f'TimeSpent(function: {func.__name__[:-4]}): {cost_time}s\n')
                return result
            return func(*args, **kwargs)
        return _wrapper

    @staticmethod
    def get_headers(host_url: str,
                    if_api: bool = False,
                    if_referer_for_host: bool = True,
                    if_ajax_for_api: bool = True,
                    if_json_for_api: bool = False,
                    if_multipart_for_api: bool = False,
                    if_http_override_for_api: bool = False
                    ) -> dict:

        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
        host_headers = {
            'Referer' if if_referer_for_host else 'Host': host_url,
            "User-Agent": user_agent,
        }
        api_headers = {
            'Origin': f'https://{urllib.parse.urlparse(host_url.strip("/")).netloc}',
            'Referer': host_url,
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            "User-Agent": user_agent,
        }
        if if_api and not if_ajax_for_api:
            api_headers.pop('X-Requested-With')
            api_headers.update({'Content-Type': 'text/plain'})
        if if_api and if_json_for_api:
            api_headers.update({'Content-Type': 'application/json'})
        if if_api and if_multipart_for_api:
            api_headers.pop('Content-Type')
        if if_api and if_http_override_for_api:
            api_headers.update({'X-HTTP-Method-Override': 'GET'})
        return host_headers if not if_api else api_headers

    def check_en_lang(self, from_lang: str, to_lang: str, default_translator: Optional[str] = None, default_lang: str = 'en-US') -> Tuple[str, str]:
        if default_translator and default_translator in self.transform_en_translator_pool:
            from_lang = default_lang if from_lang == 'en' else from_lang
            to_lang = default_lang if to_lang == 'en' else to_lang
            from_lang = default_lang.replace(
                '-', '_') if default_translator == 'lingvanex' and from_lang[:3] == 'en-' else from_lang
            to_lang = default_lang.replace(
                '-', '_') if default_translator == 'lingvanex' and to_lang[:3] == 'en-' else to_lang
        return from_lang, to_lang

    def check_language(self,
                       from_language: str,
                       to_language: str,
                       language_map: dict,
                       output_auto: str = 'auto',
                       output_zh: str = 'zh',
                       output_en_translator: Optional[str] = None,
                       output_en: str = 'en-US',
                       if_check_lang_reverse: bool = True,
                       ) -> Tuple[str, str]:

        if output_en_translator:
            from_language, to_language = self.check_en_lang(
                from_language, to_language, output_en_translator, output_en)

        from_language = output_auto if from_language in self.auto_pool else from_language
        from_language = output_zh if from_language in self.zh_pool else from_language
        to_language = output_zh if to_language in self.zh_pool else to_language

        if from_language != output_auto and from_language not in language_map:
            raise AlibabaTranslateError('Unsupported from_language[{}] in {}.'.format(
                from_language, sorted(language_map.keys())))
        elif to_language not in language_map and if_check_lang_reverse:
            raise AlibabaTranslateError('Unsupported to_language[{}] in {}.'.format(
                to_language, sorted(language_map.keys())))
        elif from_language != output_auto and to_language not in language_map[from_language]:
            raise AlibabaTranslateError('Unsupported translation: from [{0}] to [{1}]!'.format(
                from_language, to_language))
        elif from_language == to_language:
            raise AlibabaTranslateError(
                f'from_language[{from_language}] and to_language[{to_language}] should not be same.')
        return from_language, to_language

    @staticmethod
    def warning_auto_lang(translator: str, default_from_language: str, if_print_warning: bool = True) -> str:
        if if_print_warning:
            warn_tips = f'Unsupported [from_language=auto({default_from_language} instead)] with [{translator}]!'
            warnings.warn(f'{warn_tips} Please specify it.')
        return default_from_language

    @staticmethod
    def debug_lang_kwargs(from_language: str, to_language: str, default_from_language: str, if_print_warning: bool = True) -> dict:
        kwargs = {
            'from_language': from_language,
            'to_language': to_language,
            'default_from_language': default_from_language,
            'if_print_warning': if_print_warning,
        }
        return kwargs

    @staticmethod
    def debug_language_map(func):
        def make_temp_language_map(from_language: str, to_language: str, default_from_language: str) -> dict:
            if from_language == to_language or to_language == 'auto':
                raise AlibabaTranslateError

            temp_language_map = {from_language: to_language}
            if from_language != 'auto':
                temp_language_map.update({to_language: from_language})
            elif default_from_language != to_language:
                temp_language_map.update(
                    {default_from_language: to_language, to_language: default_from_language})

            return temp_language_map

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                language_map = func(*args, **kwargs)
                if not language_map:
                    raise AlibabaTranslateError
                return language_map
            except Exception as e:
                if kwargs.get('if_print_warning', True):
                    warnings.warn(
                        f'GetLanguageMapError: {str(e)}.\nThe function make_temp_language_map() works.')
                return make_temp_language_map(kwargs.get('from_language'), kwargs.get('to_language'), kwargs.get('default_from_language'))
        return _wrapper

    @staticmethod
    def check_input_limit(query_text: str, input_limit: int) -> None:
        if len(query_text) > input_limit:
            raise AlibabaTranslateError

    @staticmethod
    def check_query(func):
        def check_query_text(query_text: str, if_ignore_empty_query: bool, if_ignore_limit_of_length: bool, limit_of_length: int, bias_of_length: int = 10) -> str:
            if not isinstance(query_text, str):
                raise AlibabaTranslateError

            query_text = query_text.strip()
            qt_length = len(query_text)
            limit_of_length -= bias_of_length  # #154

            if qt_length == 0 and not if_ignore_empty_query:
                raise AlibabaTranslateError("The `query_text` can't be empty!")
            if qt_length >= limit_of_length and not if_ignore_limit_of_length:
                raise AlibabaTranslateError(
                    'The length of `query_text` exceeds the limit.')
            else:
                if qt_length >= limit_of_length:
                    warnings.warn(
                        f'The length of `query_text` is {qt_length}, above {limit_of_length}.')
                    return query_text[:limit_of_length]
            return query_text

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if_ignore_empty_query = kwargs.get('if_ignore_empty_query', True)
            if_ignore_limit_of_length = kwargs.get(
                'if_ignore_limit_of_length', False)
            limit_of_length = kwargs.get('limit_of_length', 20000)
            is_detail_result = kwargs.get('is_detail_result', False)

            query_text = list(args)[1] if len(
                args) >= 2 else kwargs.get('query_text')
            query_text = check_query_text(
                query_text, if_ignore_empty_query, if_ignore_limit_of_length, limit_of_length)
            if not query_text and if_ignore_empty_query:
                return {'data': query_text} if is_detail_result else query_text

            if len(args) >= 2:
                new_args = list(args)
                new_args[1] = query_text
                return func(*tuple(new_args), **kwargs)
            return func(*args, **{**kwargs, **{'query_text': query_text}})
        return _wrapper

    @staticmethod
    def get_client_session(http_client: str = 'requests', proxies: Optional[dict] = None) -> SessionType:
        if http_client not in ('requests', 'niquests', 'httpx', 'cloudscraper'):
            raise AlibabaTranslateError

        if proxies is None:
            proxies = {}
        session = requests.Session()
        session.proxies = proxies
        return session

    def __init__(self):
        # merged Tse init
        self.all_begin_time = time.time()
        self.default_session_freq = int(1e3)
        self.default_session_seconds = 1.5e3
        self.transform_en_translator_pool = (
            'itranslate', 'lingvanex', 'myMemory', 'apertium', 'cloudTranslation', 'translateMe')
        self.auto_pool = ('auto', 'detect', 'auto-detect', 'all')
        self.zh_pool = ('zh', 'zh-CN', 'zh-cn', 'zh-CHS',
                        'zh-Hans', 'zh-Hans_CN', 'cn', 'chi', 'Chinese')

        # original AlibabaV2 init
        self.begin_time = time.time()
        self.host_url = 'https://translate.alibaba.com'
        self.api_url = 'https://translate.alibaba.com/api/translate/text'
        self.csrf_url = 'https://translate.alibaba.com/api/translate/csrftoken'
        self.get_language_pattern = '//lang.alicdn.com/mcms/translation-open-portal/(.*?)/translation-open-portal_interface.json'
        self.get_language_url = None
        self.host_headers = self.get_headers(self.host_url, if_api=False)
        self.api_headers = self.get_headers(
            self.host_url, if_api=True, if_ajax_for_api=False, if_multipart_for_api=True)
        self.language_map = None
        self.detail_language_map = None
        self.professional_field = ('general',)
        self.csrf_token = None
        self.session = None
        self.query_count = 0
        self.output_zh = 'zh'
        self.input_limit = int(5e3)
        self.default_from_language = self.output_zh

    @debug_language_map
    def get_language_map(self, lang_html: str, **kwargs: LangMapKwargsType) -> dict:
        lang_paragraph = re.compile(
            '"en_US":{(.*?)},"zh_CN":{').search(lang_html).group().replace('",', '",\n')
        lang_items = re.compile(
            'interface.(.*?)":"(.*?)"').findall(lang_paragraph)

        def _fn_filter(k, v): return 1 if (len(k) <= 3 or (
            len(k) == 5 and '-' in k)) and len(v.split(' ')) <= 2 else 0
        lang_items = sorted([(k, v)
                            for k, v in lang_items if _fn_filter(k, v)])
        d_lang_map = {k: v for k, v in lang_items}
        lang_list = list(d_lang_map.keys())
        return {}.fromkeys(lang_list, lang_list)

    def get_d_lang_map(self, lang_html: str) -> dict:
        lang_paragraph = re.compile(
            '"en_US":{(.*?)},"zh_CN":{').search(lang_html).group().replace('",', '",\n')
        lang_items = re.compile(
            'interface.(.*?)":"(.*?)"').findall(lang_paragraph)

        def _fn_filter(k, v): return 1 if (len(k) <= 3 or (
            len(k) == 5 and '-' in k)) and len(v.split(' ')) <= 2 else 0
        lang_items = sorted([(k, v)
                            for k, v in lang_items if _fn_filter(k, v)])
        return {k: v for k, v in lang_items}

    @time_stat
    @check_query
    def alibaba_api(self, query_text: str, from_language: str = 'auto', to_language: str = 'en', **kwargs: ApiKwargsType) -> Union[str, dict]:
        """
        https://translate.alibaba.com
        :param query_text: str, must.
        :param from_language: str, default 'auto'.
        :param to_language: str, default 'en'.
        :param **kwargs:
                :param timeout: Optional[float], default None.
                :param proxies: Optional[dict], default None.
                :param sleep_seconds: float, default 0.
                :param is_detail_result: bool, default False.
                :param http_client: str, default 'requests'. Union['requests', 'niquests', 'httpx', 'cloudscraper']
                :param if_ignore_limit_of_length: bool, default False.
                :param limit_of_length: int, default 20000.
                :param if_ignore_empty_query: bool, default False.
                :param update_session_after_freq: int, default 1000.
                :param update_session_after_seconds: float, default 1500.
                :param if_show_time_stat: bool, default False.
                :param show_time_stat_precision: int, default 2.
                :param if_print_warning: bool, default True.
                :param professional_field: str, default 'message', choose from ("general",)
        :return: str or dict
        """

        use_domain = kwargs.get('professional_field', 'general')
        if use_domain not in self.professional_field:
            raise AlibabaTranslateError

        timeout = kwargs.get('timeout', None)
        proxies = kwargs.get('proxies', None)
        sleep_seconds = kwargs.get('sleep_seconds', 0)
        http_client = kwargs.get('http_client', 'requests')
        if_print_warning = kwargs.get('if_print_warning', True)
        is_detail_result = kwargs.get('is_detail_result', False)
        update_session_after_freq = kwargs.get(
            'update_session_after_freq', self.default_session_freq)
        update_session_after_seconds = kwargs.get(
            'update_session_after_seconds', self.default_session_seconds)
        self.check_input_limit(query_text, self.input_limit)

        not_update_cond_freq = 1 if self.query_count % update_session_after_freq != 0 else 0
        not_update_cond_time = 1 if time.time(
        ) - self.begin_time < update_session_after_seconds else 0
        if not (self.session and self.language_map and not_update_cond_freq and not_update_cond_time and self.csrf_token):
            self.begin_time = time.time()
            self.session = self.get_client_session(http_client, proxies)
            host_html = self.session.get(
                self.host_url, headers=self.host_headers, timeout=timeout).text
            self.get_language_url = f'https:{re.compile(self.get_language_pattern).search(host_html).group()}'
            lang_html = self.session.get(
                self.get_language_url, headers=self.host_headers, timeout=timeout).text
            debug_lang_kwargs = self.debug_lang_kwargs(
                from_language, to_language, self.default_from_language, if_print_warning)
            self.language_map = self.get_language_map(
                lang_html, **debug_lang_kwargs)
            self.detail_language_map = self.get_d_lang_map(lang_html)

            _ = self.session.get(
                self.csrf_url, headers=self.host_headers, timeout=timeout)
            self.csrf_token = self.session.get(
                self.csrf_url, headers=self.host_headers, timeout=timeout).json()
            self.api_headers.update(
                {self.csrf_token['headerName']: self.csrf_token['token']})

        from_language, to_language = self.check_language(
            from_language, to_language, self.language_map, self.output_zh)
        files_data = {
            'query': (None, query_text),
            'srcLang': (None, from_language),
            'tgtLang': (None, to_language),
            '_csrf': (None, self.csrf_token['token']),
            'domain': (None, self.professional_field[0]),
        }  # Content-Type: multipart/form-data
        r = self.session.post(self.api_url, files=files_data,
                              headers=self.api_headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        time.sleep(sleep_seconds)
        self.query_count += 1
        return data if is_detail_result else data['data']['translateText']
