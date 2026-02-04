import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';

import { defineConfig } from 'vite';

// 读取package.json文件
const packageJson = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf-8'));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  define: {
    __VERSION__: JSON.stringify(packageJson.version)
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 5173,
    open: true,
    fs: {
      allow: ['..'] // 允许访问上级目录的文件
    }
  }
});