## TTS-total

`TTS-total` 是当前仓库里的一个新 TTS 传递模式。

- 旧模式：通过 WebSocket 流式传输，边收边播
- 新模式：完整合成音频，服务端保存为 MP3，再把 `audio_url` 返回给前端播放

## 接口

- `POST /api/tts-total/speak`
- `GET /api/tts-total/files/{filename}`

## 请求示例

```json
{
  "text": "你好，我是猫猫。",
  "voice_type": "deep_male",
  "voice": null,
  "filename_prefix": "tts_total"
}
```

## 返回说明

- 成功后会返回 `success=true`
- `mode=total`
- `format=mp3`
- `audio_url`

## 文件保存位置

- `TTS-total/storage/`

## 前端切换方式

在 `chat_interface.html` 里找到 `TTS_OUTPUT_MODE`：

- `1` = 旧流式模式
- `2` = 新总量 MP3 模式

## 与旧模式的区别

- 旧模式：`/ws/tts`，边生成边播放
- 新模式：`/api/tts-total/speak`，完整合成后保存 MP3，再播放

## Git 忽略

生成的音频文件不会提交到 git，`TTS-total/.gitignore` 已忽略：

- `*.mp3`
- `*.wav`
- `*.pcm`
- `*.raw`
- `*.bin`
- `storage/*`

## 注意

- 如果现有 TTS 返回的不是 mp3，则会先保存原始调试文件
- 如果本机有 `ffmpeg`，会自动转换为 mp3
- 如果本机没有 `ffmpeg`，接口会返回错误并保留原始调试文件，方便排查
