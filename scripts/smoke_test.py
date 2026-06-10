#!/usr/bin/env python3
"""
Minimal smoke tests for the current diva-lab-agent runtime.

Phase 1 goal:
- do not change business logic
- verify the existing page and service entrypoints are reachable
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple


HOST = os.getenv("SMOKE_TEST_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
CHAT_PORT = int(os.getenv("CHAT_PORT", "8003"))
TTS_PORT = int(os.getenv("TTS_PORT", "8004"))
SCORING_PORT = int(os.getenv("SCORING_PORT", "8005"))
ASR_PORT = int(os.getenv("ASR_PORT", "8006"))
TIMEOUT_SEC = float(os.getenv("SMOKE_TEST_TIMEOUT", "5"))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def http_check(name: str, url: str, expected_statuses: Tuple[int, ...] = (200,)) -> CheckResult:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            status = getattr(resp, "status", 200)
            if status in expected_statuses:
                return CheckResult(name, True, f"HTTP {status} {url}")
            return CheckResult(name, False, f"unexpected HTTP {status} {url}")
    except urllib.error.HTTPError as e:
        if e.code in expected_statuses:
            return CheckResult(name, True, f"HTTP {e.code} {url}")
        return CheckResult(name, False, f"HTTPError {e.code} {url}")
    except Exception as e:
        return CheckResult(name, False, f"{type(e).__name__}: {e}")


def run_checks() -> List[CheckResult]:
    return [
        http_check("web-page-8000", f"http://{HOST}:{WEB_PORT}/tablet_legacy.html", (200,)),
        http_check("full-web-page-8000", f"http://{HOST}:{WEB_PORT}/mao_demo.html", (200,)),
        http_check("chat-health-8003", f"http://{HOST}:{CHAT_PORT}/health", (200,)),
        http_check("tts-docs-8004", f"http://{HOST}:{TTS_PORT}/docs", (200,)),
        http_check("scoring-health-8005", f"http://{HOST}:{SCORING_PORT}/health", (200,)),
        http_check("asr-health-8006", f"http://{HOST}:{ASR_PORT}/health", (200,)),
    ]


def main() -> int:
    results = run_checks()
    failed = [r for r in results if not r.ok]

    print("Smoke test results:")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"- [{status}] {result.name}: {result.detail}")

    if failed:
        print(f"\n{len(failed)} check(s) failed.")
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
