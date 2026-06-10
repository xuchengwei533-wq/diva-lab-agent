# Xiaosita Vercel Proxy

This small Vercel project is a stable public entrypoint for the tablet APK.

It does not run the Xiaosita audio/chat backend on Vercel. Instead, it forwards
all requests to the current public backend tunnel configured with
`BACKEND_BASE_URL`.

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
