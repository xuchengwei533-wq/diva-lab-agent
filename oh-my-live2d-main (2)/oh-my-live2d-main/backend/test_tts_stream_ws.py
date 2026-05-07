import json
import sys
import time

from websocket import WebSocketTimeoutException, create_connection


WS_URL = (
    "ws://127.0.0.1:8004/ws/tts"
    "?voice=Ethan&model=qwen3-tts-flash&language_type=Chinese"
)
TEST_TEXT = "你好，今天我们来练习唱歌。请保持放松。"
ALLOWED_MALE_VOICES = {"Ethan", "Eldric Sage", "Ryan", "Aiden", "Vincent", "Neil", "Andre"}


def print_message(prefix: str, payload):
    print(prefix, json.dumps(payload, ensure_ascii=False))


def main() -> int:
    delta_count = 0
    saw_url = False
    saw_base64 = False
    seen_audio_voices = []
    errors = []
    seen_types = []
    ws = None

    try:
        ws = create_connection(WS_URL, timeout=10)
        ws.settimeout(15)

        ready = json.loads(ws.recv())
        seen_types.append(ready.get("type"))
        print_message("recv", {"type": ready.get("type"), "sample_rate": ready.get("sample_rate"), "format": ready.get("format")})

        ws.send(json.dumps({"type": "input_text_buffer.append", "text": TEST_TEXT}, ensure_ascii=False))
        ws.send(json.dumps({"type": "input_text_buffer.commit"}, ensure_ascii=False))

        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                print("recv timeout")
                break

            if not raw:
                continue

            msg = json.loads(raw)
            mtype = msg.get("type", "<unknown>")
            seen_types.append(mtype)
            print("recv", mtype)

            if mtype == "response.audio.delta":
                delta_count += 1
                voice = msg.get("voice")
                if voice:
                    seen_audio_voices.append(voice)
                if delta_count <= 3:
                    print_message(
                        "delta_meta",
                        {
                            "count": delta_count,
                            "format": msg.get("format"),
                            "sample_rate": msg.get("sample_rate"),
                            "delta_len": len(msg.get("delta") or ""),
                            "provider": msg.get("provider"),
                            "model": msg.get("model"),
                            "voice": msg.get("voice"),
                        },
                    )
            elif mtype == "response.audio.url":
                saw_url = True
                voice = msg.get("voice")
                if voice:
                    seen_audio_voices.append(voice)
                print_message(
                    "fallback_url",
                    {
                        "format": msg.get("format"),
                        "provider": msg.get("provider"),
                        "model": msg.get("model"),
                        "voice": msg.get("voice"),
                        "url": msg.get("url"),
                    },
                )
            elif mtype == "response.audio.base64":
                saw_base64 = True
                voice = msg.get("voice")
                if voice:
                    seen_audio_voices.append(voice)
                print_message(
                    "fallback_base64",
                    {
                        "format": msg.get("format"),
                        "provider": msg.get("provider"),
                        "model": msg.get("model"),
                        "voice": msg.get("voice"),
                        "audio_base64_len": len(msg.get("audio_base64") or ""),
                    },
                )
            elif mtype == "error":
                errors.append(msg)
                print_message(
                    "error_detail",
                    {
                        "error": msg.get("error"),
                        "message": msg.get("message"),
                        "attempts": msg.get("attempts") or [],
                    },
                )
            elif mtype == "response.done":
                break

        summary = {
            "delta_count": delta_count,
            "saw_url": saw_url,
            "saw_base64": saw_base64,
            "seen_audio_voices": seen_audio_voices,
            "error_count": len(errors),
            "seen_types": seen_types,
        }
        print_message("summary", summary)

        if delta_count >= 2:
            invalid_voices = [v for v in seen_audio_voices if v not in ALLOWED_MALE_VOICES]
            if invalid_voices:
                print_message("invalid_voices", {"voices": invalid_voices})
                print("VERDICT: 流式已生效，但默认音色不是目标男声")
                return 3
            print("VERDICT: 真正流式 TTS（收到至少 2 个 response.audio.delta）")
            return 0
        if saw_url or saw_base64:
            invalid_voices = [v for v in seen_audio_voices if v not in ALLOWED_MALE_VOICES]
            if invalid_voices:
                print_message("invalid_voices", {"voices": invalid_voices})
                print("VERDICT: 非流式 fallback，且音色不是目标男声")
                return 3
            print("VERDICT: 非流式 fallback（收到 response.audio.url 或 response.audio.base64）")
            if errors:
                print_message("fallback_reason", {"attempts": errors[-1].get("attempts") or []})
            return 0
        if errors:
            print("VERDICT: 仅收到 error")
            print_message("attempts", errors[-1].get("attempts") or [])
            return 1

        print("VERDICT: 未收到可判定的 TTS 音频消息")
        return 2
    finally:
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "session.finish"}, ensure_ascii=False))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
