# Legacy Android Tablet Deployment

This build targets old, low-power Android tablets first.

## Supported Baseline

- Minimum APK install target: Android 5.0 / API 21.
- Tablets below Android 5.0 are not a reliable target for this project because WebView media permissions and modern WebView APIs are incomplete.
- On Android 5.0 and newer, update Android System WebView or Chrome to the newest version that the tablet can install before testing.

## Default Tablet URL

The APK and startup scripts now use the lightweight URL by default:

```text
http://192.168.5.140:8000/tablet_legacy.html?apiHost=192.168.5.140&live2dHost=192.168.5.140&live2dPort=8010&legacy=1&lite=1&renderMode=gif&disableFace=1
```

Replace `192.168.5.140` with the server computer's LAN IP if it changes.

## Lightweight Mode

These query parameters are passed from the home page to the chat page:

- `legacy=1` / `lite=1`: prefer low-power behavior.
- `renderMode=gif`: skip Live2D and use local GIF character rendering.
- `disableFace=1`: skip face-api, face model downloads, and camera emotion tracking.

The chat, ASR, TTS, scoring, and static asset services still run on the server computer. The tablet only renders the WebView shell and sends requests over the LAN.

For very old WebView builds, `tablet_legacy.html` is the safest entry. It avoids modern JavaScript syntax, `fetch`, Live2D, face-api, charts, and camera emotion analysis.

## Start Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_tablet_server.ps1 -HostIp 192.168.5.140
```

Stop it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_tablet_server.ps1 -Stop
```
