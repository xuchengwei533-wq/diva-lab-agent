import os

import dashscope


def _load_env_file():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, '.env'),
        os.path.join(os.path.dirname(here), '.env'),
        os.path.join(os.path.dirname(os.path.dirname(here)), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass
        break


_load_env_file()

dashscope.base_http_api_url = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

DEFAULT_MODEL = os.getenv('DASHSCOPE_TTS_MODEL', 'qwen3-tts-flash')
DEFAULT_VOICE = os.getenv('DASHSCOPE_TTS_VOICE', 'Ethan')
DEFAULT_LANG = os.getenv('DASHSCOPE_TTS_LANG', 'Chinese')
MIN_CHARS_PER_REQ = int(os.getenv('TTS_MIN_CHARS_PER_REQ', '40'))
MAX_CHARS_PER_REQ = int(os.getenv('TTS_MAX_CHARS_PER_REQ', '220'))
MIN_TTS_MERGE_CHARS = int(os.getenv('TTS_MIN_MERGE_CHARS', '15'))
PCM_SAMPLE_RATE = int(os.getenv('TTS_PCM_SAMPLE_RATE', '24000'))
PCM_FORMAT = os.getenv('TTS_PCM_FORMAT', 'pcm_s16le')
CHINESE_PROBE_TEXT = os.getenv('TTS_PROBE_TEXT_ZH', '你好，今天我们来练习唱歌。')
DEFAULT_PROBE_TEXT = os.getenv('TTS_PROBE_TEXT_DEFAULT', 'Hello, today we will practice singing.')
