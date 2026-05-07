import json
import time

from websocket import WebSocketTimeoutException, create_connection

from tts_ws_server import MIN_TTS_MERGE_CHARS, _split_text_for_tts


WS_URL = (
    "ws://127.0.0.1:8004/ws/tts"
    "?voice=Ethan&model=qwen3-tts-flash&language_type=Chinese"
)
TEST_TEXT = (
    "很高兴听见你这么说，今天我们慢慢来，不着急。"
    "先把呼吸放稳，再让声音自然地向前走。"
)


def print_json(prefix: str, payload) -> None:
    print(prefix, json.dumps(payload, ensure_ascii=False))


def validate_segments() -> bool:
    segments = _split_text_for_tts(TEST_TEXT)
    print_json("TTS_SEGMENT", [{"len": len(seg), "text": seg} for seg in segments])

    ok = True
    for seg in segments:
        if len(seg) < MIN_TTS_MERGE_CHARS and len(TEST_TEXT) >= MIN_TTS_MERGE_CHARS:
            print(f"ERROR: 出现过短 segment: len={len(seg)} text={seg}")
            ok = False

    joined = "|".join(segments)
    if "听见你这么说" not in joined:
        print("ERROR: “听见你这么说” 被错误切开了")
        ok = False

    return ok


def run_ws_check() -> int:
    delta_count = 0
    saw_url = False
    saw_base64 = False
    errors = []
    ws = None

    try:
        ws = create_connection(WS_URL, timeout=10)
        ws.settimeout(20)

        ready = json.loads(ws.recv())
        print_json(
            "session.ready",
            {
                "type": ready.get("type"),
                "format": ready.get("format"),
                "sample_rate": ready.get("sample_rate"),
                "voice": ready.get("voice"),
            },
        )

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
            mtype = msg.get("type")
            print("recv", mtype)

            if mtype == "response.audio.delta":
                delta_count += 1
                print_json(
                    "delta_meta",
                    {
                        "count": delta_count,
                        "format": msg.get("format"),
                        "sample_rate": msg.get("sample_rate"),
                        "voice": msg.get("voice"),
                        "model": msg.get("model"),
                        "delta_len": len(msg.get("delta") or ""),
                    },
                )
            elif mtype == "response.audio.url":
                saw_url = True
                print_json(
                    "fallback_url",
                    {
                        "format": msg.get("format"),
                        "voice": msg.get("voice"),
                        "model": msg.get("model"),
                        "url": msg.get("url"),
                    },
                )
            elif mtype == "response.audio.base64":
                saw_base64 = True
                print_json(
                    "fallback_base64",
                    {
                        "format": msg.get("format"),
                        "voice": msg.get("voice"),
                        "model": msg.get("model"),
                        "audio_base64_len": len(msg.get("audio_base64") or ""),
                    },
                )
            elif mtype == "error":
                errors.append(msg)
                print_json(
                    "error_detail",
                    {
                        "error": msg.get("error"),
                        "message": msg.get("message"),
                        "attempts": msg.get("attempts") or [],
                    },
                )
            elif mtype == "response.done":
                break

        print_json(
            "summary",
            {
                "delta_count": delta_count,
                "saw_url": saw_url,
                "saw_base64": saw_base64,
                "error_count": len(errors),
            },
        )

        if delta_count >= 2:
            print("VERDICT: 流式 TTS 正常，且收到了多个 response.audio.delta")
            return 0
        if saw_url or saw_base64:
            print("VERDICT: 触发了 fallback")
            if errors:
                print_json("fallback_reason", {"attempts": errors[-1].get("attempts") or []})
            return 0
        if errors:
            print("VERDICT: 仅收到 error")
            print_json("attempts", errors[-1].get("attempts") or [])
            return 1

        print("VERDICT: 未收到可判定的音频消息")
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


def main() -> int:
    print("=== split_text_for_tts ===")
    ok = validate_segments()
    print("=== websocket ===")
    result = run_ws_check()
    if not ok and result == 0:
        return 3
    return result if result != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
