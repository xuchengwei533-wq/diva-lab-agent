# Android Tablet Deployment

This project should be deployed to Android as a thin WebView APK.

Do not package the Python backend, PyTorch models, DashScope clients, or Live2D asset server into the APK. Keep those services on a Windows PC, LAN server, or cloud server. The APK only opens the web UI and connects to the backend host.

## What Goes Into The APK

- Android WebView shell from `android-tablet/`
- Runtime permissions for microphone and camera
- A small address input so the tablet can point to a backend host

## What Stays On The Backend Machine

- `mao_demo_server.py` page and static asset service
- `backend/qwen_chat_server.py`
- `backend/tts_ws_server.py`
- `backend/asr_server.py`
- `backend/asr_new.py`
- PyTorch model weights and Python dependencies

## Start The Backend

On the backend PC:

```powershell
conda activate diva-lab
cd "C:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main\backend"
.\start_all_servers.bat
```

Print the tablet URL:

```powershell
cd "C:\Project\Mao-zhishi"
python scripts\tablet_url.py
```

Example:

```text
http://192.168.1.23:8000/mao_demo.html?apiHost=192.168.1.23&live2dHost=192.168.1.23&live2dPort=8010
```

Open that URL in the Android tablet browser first. If it works in the browser, the APK will use the same backend address.

## Build The APK

1. Open `C:\Project\Mao-zhishi\android-tablet` in Android Studio.
2. Let Android Studio sync Gradle.
3. Build a debug APK with `Build > Build Bundle(s) / APK(s) > Build APK(s)`.
4. Install the APK on the tablet.
5. In the APK address field, enter the backend PC IP, for example:

```text
192.168.1.23
```

The APK will load:

```text
http://192.168.1.23:8000/mao_demo.html?apiHost=192.168.1.23&live2dHost=192.168.1.23&live2dPort=8010
```

## Network Notes

- The tablet and backend PC must be on the same Wi-Fi/LAN, unless the backend is deployed to a public server.
- Windows Firewall must allow inbound access to ports `8000`, `8001`, `8002`, `8003`, `8004`, `8005`, `8006`, and `8010`.
- For production, put the backend behind HTTPS and use `apiProtocol=https&wsProtocol=wss`.

## URL Parameters

The web pages now support these deployment parameters:

- `apiHost`: backend host for chat, TTS, ASR, and scoring
- `apiProtocol`: `http` or `https`
- `wsProtocol`: `ws` or `wss`
- `chatPort`: default `8003`
- `ttsPort`: default `8004`
- `scoringPort`: default `8005`
- `asrPort`: default `8006`
- `live2dHost`: host for Live2D/static assets
- `live2dPort`: default `8010` when remote assets are used
- `live2dBase`: full Live2D asset base URL override

