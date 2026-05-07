import asyncio
import json

import tts_ws_server as tts


TEST_TEXT = tts.CHINESE_PROBE_TEXT


async def main():
    print(f"Testing text: {TEST_TEXT}")
    print(f"DashScope base url: {tts.dashscope.base_http_api_url}")
    print(f"Candidate count: {len(tts.TTS_CANDIDATES)}")
    print("")

    attempts = []
    success = None

    for candidate in tts.TTS_CANDIDATES:
        meta, attempt = await tts._call_tts_candidate(TEST_TEXT, candidate)
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False))
        if meta:
            success = {
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "voice": meta.get("voice"),
                "language_type": meta.get("language_type"),
                "format": meta.get("format"),
                "has_audio_url": bool(meta.get("audio_url")),
                "has_audio_base64": bool(meta.get("audio_base64")),
            }
            break

    print("")
    if success:
        print(
            "SUCCESS "
            f"provider={success['provider']} "
            f"model={success['model']} "
            f"voice={success['voice']} "
            f"language_type={success['language_type']} "
            f"format={success['format']} "
            f"has_audio_url={success['has_audio_url']} "
            f"has_audio_base64={success['has_audio_base64']}"
        )
    else:
        print("All candidates failed.")
        print(json.dumps({"attempts": attempts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
