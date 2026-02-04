#!/usr/bin/env python3
"""
mao-demo HTTP服务器
在8000端口提供mao-demo.html预览
"""

import http.server
import socketserver
import os
import sys

# 设置项目根目录为工作目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

PORT = 8000

class MaoDemoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
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
    with socketserver.TCPServer(("", PORT), MaoDemoHTTPRequestHandler) as httpd:
        print(f"mao-demo服务器启动成功!")
        print(f"访问地址: http://localhost:{PORT}/mao_demo.html")
        print(f"服务目录: {PROJECT_ROOT}")
        print("按 Ctrl+C 停止服务器")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    main()