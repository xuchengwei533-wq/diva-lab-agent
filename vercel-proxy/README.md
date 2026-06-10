# Xiaosita Vercel Proxy

This small Vercel project is a stable public entrypoint for the tablet APK.

It does not run the Xiaosita audio/chat backend on Vercel. Instead, it forwards
all requests to the current public backend tunnel configured with
`BACKEND_BASE_URL`.

## Current Production URL

```text
https://vercel-proxy-nine-beta.vercel.app
```

## Required Environment Variable

```text
BACKEND_BASE_URL=https://your-current-public-backend.example.com
```

For the current local deployment, this value should point to the active
Cloudflare Tunnel URL, for example:

```text
BACKEND_BASE_URL=https://reporters-dealtime-defence-advised.trycloudflare.com
```

## Paths

All paths are forwarded:

- `/tablet_legacy.html`
- `/packages/...`
- `/api/chat`
- `/api/tts/...`
- `/api/tts-total/...`
- `/api/voice/start`
- `/proxy-audio?...`

The Android APK should point at the final Vercel URL, not directly at the
temporary Cloudflare Tunnel URL.

## One-Command Refresh

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_tablet_public.ps1
```

The script starts the local backend, creates a fresh free Cloudflare Tunnel,
updates Vercel's Production `BACKEND_BASE_URL`, redeploys the proxy, and checks
the final tablet URL.

Useful options:

```powershell
# Reuse the existing live tunnel when it is still healthy.
powershell -ExecutionPolicy Bypass -File scripts\deploy_tablet_public.ps1 -ReuseTunnel

# Skip backend restart and only refresh the tunnel/Vercel side.
powershell -ExecutionPolicy Bypass -File scripts\deploy_tablet_public.ps1 -SkipBackendRestart

# Also run a real /api/chat smoke test after deployment.
powershell -ExecutionPolicy Bypass -File scripts\deploy_tablet_public.ps1 -RunChatSmoke
```

The latest result is written to `logs\tablet-server\deploy_status.txt`.
