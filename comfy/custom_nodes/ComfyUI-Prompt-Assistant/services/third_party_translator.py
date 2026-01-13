
import random
import time
import asyncio
import warnings
from functools import lru_cache

# Suppress 'Unable to find server backend' warning from translators library
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Unable to find server backend.*")
    import translators as ts

from ..utils.common import ProgressBar, log_prepare, log_error, TASK_TRANSLATE, SOURCE_NODE

# Extracted from enhanced/translator.py
translator_org = ['alibaba','cloudTranslation', 'google', 'iflyrec', 'translateCom', 'youdao']
translator_default = 'alibaba'

@lru_cache(maxsize=32, typed=False)
def translate2zh_apis(text):
    global translator_default
    if not text:
        return text
    try:
        return ts.translate_text(text, translator=translator_default, from_language='en', to_language='zh')
    except Exception as e:
        try:
            # logger.info(f'Change another translator because of {e}')
            translator_default = translator_org[random.randint(0,5)]
            return ts.translate_text(text, translator=translator_default, from_language='en', to_language='zh')
        except Exception as e:
            # logger.info(f'Error during translation of APIs methods: {e}')
            raise e

def translate2en_apis(text):
    global translator_default
    if not text:
        return text
    try:
        return ts.translate_text(text, translator=translator_default, to_language='en')
    except Exception as e:
        try:
            # logger.info(f'Change another translator because of {e}')
            translator_default = translator_org[random.randint(0,5)]
            return ts.translate_text(text, translator=translator_default, to_language='en')
        except Exception as e:
            # logger.info(f'Error during translation of APIs methods: {e}')
            raise e

class ThirdPartyTranslateService:
    @staticmethod
    async def translate(text, from_lang='auto', to_lang='zh', request_id=None, cancel_event=None, task_type=None, source=None, **kwargs):
        """
        Use Third-party APIs for translation (extracted from enhanced/translator.py)
        """
        request_id = request_id or f"third_api_{int(time.time())}_{random.randint(1000, 9999)}"
        
        if not text or text.strip() == '':
            return {"success": False, "error": "Input text cannot be empty"}

        task_type = task_type or TASK_TRANSLATE
        
        pbar = ProgressBar(
            request_id=request_id,
            service_name="Third-party APIs",
            streaming=False,
            extra_info=f"Length:{len(text)}",
            task_type=task_type,
            source=source
        )

        try:
            loop = asyncio.get_running_loop()
            
            if cancel_event and cancel_event.is_set():
                return {"success": False, "error": "Task cancelled"}

            # Select function based on target language
            # Note: The original logic hardcoded 'en'->'zh' or 'zh'->'en'.
            # Here we adapt it to standard from_lang/to_lang
            
            def do_translate():
                if to_lang == 'zh':
                    # Assuming source is English or mixed, trying to translate to Chinese
                    return translate2zh_apis(text)
                else:
                    # Assuming target is English
                    return translate2en_apis(text)

            start_time = time.perf_counter()
            
            translated_text = await loop.run_in_executor(
                None, 
                do_translate
            )
            
            if cancel_event and cancel_event.is_set():
                return {"success": False, "error": "Task cancelled"}

            return {
                "success": True, 
                "data": {
                    "translated": translated_text,
                    "from_lang": from_lang,
                    "to_lang": to_lang
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
