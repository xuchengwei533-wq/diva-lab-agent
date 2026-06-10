#!/usr/bin/env python3
"""
mao-demo HTTP服务器
在8000端口提供mao-demo.html预览
"""

import http.server
import socketserver
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

# 设置项目根目录为工作目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

PORT = int(os.environ.get("WEB_PORT", "8000"))
MODE = (os.environ.get("WEB_MODE", "all") or "all").strip().lower()

REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, os.pardir))
FACE_API_MODELS_DIR = os.path.join(REPO_ROOT, "demo", "demo", "models")
NATORI_MODEL_DIR = os.path.join(REPO_ROOT, "natori_pro_zh")
PACKAGES_DIR = os.path.join(PROJECT_ROOT, "packages")
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor")
PROXY_TARGETS = (
    ("/api/chat", "http://127.0.0.1:8003"),
    ("/api/tts", "http://127.0.0.1:8004"),
    ("/api/tts-total", "http://127.0.0.1:8004"),
    ("/api/audio", "http://127.0.0.1:8005"),
    ("/api/voice", "http://127.0.0.1:8006"),
    ("/api/asr", "http://127.0.0.1:8006"),
)

class MaoDemoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".json": "application/json",
    }

    def do_GET(self):
        raw_path = (self.path or "/")
        path = raw_path.split("?", 1)[0].split("#", 1)[0]

        if path == "/proxy-audio":
            self.proxy_external_audio(raw_path)
            return

        proxy_base = self.get_proxy_base(path)
        if proxy_base:
            self.proxy_request(proxy_base)
            return

        if MODE != "assets":
            if path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/tablet_legacy.html")
                self.end_headers()
                return
            if path.lower() == "/mao-demo.html":
                self.send_response(302)
                self.send_header("Location", "/mao_demo.html")
                self.end_headers()
                return

        if MODE == "page":
            if (
                path not in ("/mao_demo.html", "/chat_interface.html", "/tablet_legacy.html", "/favicon.ico")
                and not path.startswith("/face-api-models/")
                and not path.startswith("/packages/")
                and not path.startswith("/natori_pro_zh/")
                and not path.startswith("/vendor/")
            ):
                self.send_error(404)
                return
        elif MODE == "assets":
            if path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/packages/")
                self.end_headers()
                return
            if not (
                path.startswith("/face-api-models/")
                or path.startswith("/packages/")
                or path.startswith("/natori_pro_zh/")
                or path.startswith("/vendor/")
            ):
                self.send_error(404)
                return

        return super().do_GET()

    def do_POST(self):
        path = (self.path or "/").split("?", 1)[0].split("#", 1)[0]
        proxy_base = self.get_proxy_base(path)
        if proxy_base:
            self.proxy_request(proxy_base)
            return
        self.send_error(404)

    def translate_path(self, path):
        raw_path = (path or "/")
        clean_path = raw_path.split("?", 1)[0].split("#", 1)[0]
        if clean_path.startswith("/face-api-models/"):
            rel = unquote(clean_path[len("/face-api-models/"):])
            rel = rel.replace("\\", "/")
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                return os.path.join(FACE_API_MODELS_DIR, "__invalid__")
            return os.path.join(FACE_API_MODELS_DIR, rel)
        if clean_path.startswith("/natori_pro_zh/"):
            rel = unquote(clean_path[len("/natori_pro_zh/"):])
            rel = rel.replace("\\", "/")
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                return os.path.join(NATORI_MODEL_DIR, "__invalid__")
            return os.path.join(NATORI_MODEL_DIR, rel)
        if clean_path.startswith("/packages/"):
            rel = unquote(clean_path[len("/packages/"):])
            rel = rel.replace("\\", "/")
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                return os.path.join(PACKAGES_DIR, "__invalid__")
            return os.path.join(PACKAGES_DIR, rel)
        if clean_path.startswith("/vendor/"):
            rel = unquote(clean_path[len("/vendor/"):])
            rel = rel.replace("\\", "/")
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                return os.path.join(VENDOR_DIR, "__invalid__")
            return os.path.join(VENDOR_DIR, rel)
        return super().translate_path(path)

    def end_headers(self):
        # 添加CORS头以允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        request_path = (self.path or "/").split("?", 1)[0].split("#", 1)[0]
        if request_path in ("/", "/index.html", "/tablet_legacy.html", "/mao_demo.html", "/chat_interface.html"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()
    
    def do_OPTIONS(self):
        # 处理预检请求
        self.send_response(200)
        self.end_headers()

    def get_proxy_base(self, path):
        if MODE == "assets":
            return None
        for prefix, target in PROXY_TARGETS:
            if path == prefix or path.startswith(prefix + "/"):
                return target
        return None

    def proxy_request(self, target_base):
        body = None
        content_length = self.headers.get("Content-Length")
        if content_length:
            try:
                body = self.rfile.read(int(content_length))
            except Exception:
                body = b""

        upstream_url = target_base.rstrip("/") + self.path
        headers = {}
        for name, value in self.headers.items():
            lname = name.lower()
            if lname in ("host", "connection", "transfer-encoding", "content-length"):
                continue
            headers[name] = value

        request = urllib.request.Request(
            upstream_url,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                self.send_response(response.status)
                for name, value in response.headers.items():
                    lname = name.lower()
                    if lname in ("connection", "transfer-encoding", "content-length"):
                        continue
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "OPTIONS":
                    self.wfile.write(data)
        except urllib.error.HTTPError as error:
            data = error.read()
            self.send_response(error.code)
            for name, value in error.headers.items():
                lname = name.lower()
                if lname in ("connection", "transfer-encoding", "content-length"):
                    continue
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "OPTIONS":
                self.wfile.write(data)
        except Exception as error:
            payload = ("Proxy request failed: " + str(error)).encode("utf-8", errors="replace")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def proxy_external_audio(self, raw_path):
        query = raw_path.split("?", 1)[1] if "?" in raw_path else ""
        audio_urls = parse_qs(query).get("url", [])
        if not audio_urls:
            self.send_error(400, "Missing audio url")
            return

        audio_url = audio_urls[0]
        parsed = urlparse(audio_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in ("http", "https") or not (
            hostname == "aliyuncs.com" or hostname.endswith(".aliyuncs.com")
        ):
            self.send_error(403, "Audio host is not allowed")
            return

        request = urllib.request.Request(
            audio_url,
            headers={"User-Agent": "Mozilla/5.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "audio/wav"))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as error:
            data = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as error:
            payload = ("Audio proxy failed: " + str(error)).encode("utf-8", errors="replace")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

def main():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), MaoDemoHTTPRequestHandler) as httpd:
        print(f"mao-demo服务器启动成功!")
        if MODE == "assets":
            print(f"资源服务: http://localhost:{PORT}/")
        else:
            print(f"访问地址: http://localhost:{PORT}/mao_demo.html")
        print(f"服务目录: {PROJECT_ROOT}")
        print(f"模式: {MODE}")
        print("按 Ctrl+C 停止服务器")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    main()
