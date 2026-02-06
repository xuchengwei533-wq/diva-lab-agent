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

# 设置项目根目录为工作目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

PORT = int(os.environ.get("WEB_PORT", "8000"))
MODE = (os.environ.get("WEB_MODE", "all") or "all").strip().lower()

REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, os.pardir))
FACE_API_MODELS_DIR = os.path.join(REPO_ROOT, "demo", "demo", "models")

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

        if MODE != "assets":
            if path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/mao_demo.html")
                self.end_headers()
                return
            if path.lower() == "/mao-demo.html":
                self.send_response(302)
                self.send_header("Location", "/mao_demo.html")
                self.end_headers()
                return

        if MODE == "page":
            if (
                path not in ("/mao_demo.html", "/chat_interface.html", "/favicon.ico")
                and not path.startswith("/face-api-models/")
                and not path.startswith("/packages/")
                and not path.startswith("/mao_pro_en/")
            ):
                self.send_error(404)
                return
        elif MODE == "assets":
            if path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/packages/")
                self.end_headers()
                return
            if not (path.startswith("/packages/") or path.startswith("/mao_pro_en/")):
                self.send_error(404)
                return

        return super().do_GET()

    def translate_path(self, path):
        raw_path = (path or "/")
        clean_path = raw_path.split("?", 1)[0].split("#", 1)[0]
        if clean_path.startswith("/face-api-models/"):
            rel = clean_path[len("/face-api-models/"):]
            rel = rel.replace("\\", "/")
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel):
                return os.path.join(FACE_API_MODELS_DIR, "__invalid__")
            return os.path.join(FACE_API_MODELS_DIR, rel)
        return super().translate_path(path)

    def end_headers(self):
        # 添加CORS头以允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # 处理预检请求
        self.send_response(200)
        self.end_headers()

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
