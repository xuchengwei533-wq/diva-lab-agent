import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def _include_tts_total_router(app: FastAPI) -> None:
    project_root = Path(__file__).resolve().parents[2]
    router_path = project_root / "TTS-total" / "router.py"
    if not router_path.is_file():
        print(f"[TTS-total] warning: router not found at {router_path}")
        return

    try:
        spec = importlib.util.spec_from_file_location("tts_total_router", router_path)
        if not spec or not spec.loader:
            raise RuntimeError("spec_from_file_location returned no loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        total_router = getattr(module, "router", None)
        if total_router is None:
            raise RuntimeError("router symbol not found")
        app.include_router(total_router)
        print(f"[TTS-total] router loaded from {router_path}")
    except Exception as exc:
        print(f"[TTS-total] warning: failed to load router from {router_path}: {exc}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router)
_include_tts_total_router(app)
