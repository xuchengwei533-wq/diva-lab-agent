from typing import Optional

from pydantic import BaseModel, Field


class TTSSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_type: Optional[str] = Field(default='deep_male')


class SessionState:
    def __init__(self):
        self.text_buf = ''
        self.closed = False
