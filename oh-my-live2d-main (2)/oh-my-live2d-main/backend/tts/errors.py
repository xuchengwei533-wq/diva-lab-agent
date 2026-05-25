from typing import Any, Dict, List


class TTSAllFailedError(RuntimeError):
    def __init__(self, message: str, attempts: List[Dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts

    def to_dict(self) -> Dict[str, Any]:
        message = self.attempts[-1].get('dashscope_message') if self.attempts else str(self)
        return {
            'type': 'error',
            'error': str(self),
            'message': message or str(self),
            'attempts': self.attempts,
        }
