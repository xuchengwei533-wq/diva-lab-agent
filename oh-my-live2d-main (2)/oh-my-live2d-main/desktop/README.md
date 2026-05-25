## Optional Desktop Pet Launcher

This directory contains an optional PySide6 desktop client for rendering a local
HTML page in a transparent, always-on-top window.

It is not part of the backend service startup flow and does not affect the
runtime services on ports `8000`, `8002`, `8003`, `8004`, `8005`, or `8006`.

### Install

```bash
cd "C:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main"
python -m pip install -r desktop/requirements.txt
```

### Run

```bash
cd "C:\Project\Mao-zhishi\oh-my-live2d-main (2)\oh-my-live2d-main"
python desktop/desktop_pet_launcher.py
```

### Optional Arguments

```bash
python desktop/desktop_pet_launcher.py --html mao_demo.html --width 420 --height 520
```
