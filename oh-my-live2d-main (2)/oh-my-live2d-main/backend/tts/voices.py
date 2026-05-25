from typing import Optional

from .config import DEFAULT_VOICE


VOICE_TYPE_TO_VOICE = {
    'default': 'Ethan',
    'male': 'Ethan',
    'deep_male': 'Ethan',
    'mature_male': 'Andre',
    'elder_male': 'Eldric Sage',
    'news_male': 'Neil',
    'calm': 'Ethan',
    'cute': 'Ethan',
    'fast': 'Ethan',
    'slow': 'Ethan',
    'mao': 'Ethan',
    'female': 'Ethan',
    'female_cute': 'Ethan',
    'female_lively': 'Neil',
    'female_gentle': 'Ethan',
}

SUPPORTED_VOICES = [
    'Ethan',
    'Eldric Sage',
    'Ryan',
    'Aiden',
    'Neil',
    'Andre',
    'Vincent',
    'Moon',
]


def normalize_voice(voice: Optional[str]) -> str:
    if not voice:
        return DEFAULT_VOICE
    value = voice.strip()
    if not value:
        return DEFAULT_VOICE
    if value in ('Cherry', 'Serena', 'Chelsie', 'cute', 'mao'):
        return DEFAULT_VOICE
    return value
