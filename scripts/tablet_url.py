#!/usr/bin/env python3
from __future__ import annotations

import os
import socket


def get_lan_ip() -> str:
    override = os.getenv("TABLET_HOST_IP")
    if override:
        return override

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> int:
    host = get_lan_ip()
    web_port = os.getenv("WEB_PORT", "8000")
    url = (
        f"http://{host}:{web_port}/tablet_legacy.html"
        "?singleOrigin=1"
        "&publicMode=1"
        "&voiceMode=1"
    )
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

