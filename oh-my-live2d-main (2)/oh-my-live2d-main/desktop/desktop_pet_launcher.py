"""
Optional desktop pet launcher for the inner oh-my-live2d-main project.

This module is a standalone PySide6 client that opens a transparent, always-on-top
window and renders a local HTML page with QWebEngineView. It is not part of the
backend service startup flow and does not affect the Web services on ports
8000/8002/8003/8004/8005/8006.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PySide6.QtCore import QEvent, QPoint, Qt, QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
except ImportError as exc:  # pragma: no cover - optional desktop dependency
    raise SystemExit(
        "PySide6 is required for the optional desktop launcher.\n"
        "Install it with: python -m pip install -r desktop/requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML = PROJECT_ROOT / "mao_demo.html"


class DesktopPet(QWidget):
    def __init__(self, html_path: Path, width: int = 420, height: int = 520):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(width, height)

        self.web = QWebEngineView(self)
        self.web.setAttribute(Qt.WA_TranslucentBackground, True)
        self.web.page().setBackgroundColor(Qt.transparent)
        self.web.installEventFilter(self)
        self.web.load(QUrl.fromLocalFile(str(html_path.resolve())))

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web)
        self.setLayout(layout)

        self._press_global = QPoint()
        self._press_win_topleft = QPoint()
        self._dragging = False
        self._drag_threshold = 6

    def eventFilter(self, obj, event):
        if obj is self.web:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._press_global = event.globalPosition().toPoint()
                self._press_win_topleft = self.frameGeometry().topLeft()
                self._dragging = False
                return False

            if event.type() == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
                cur = event.globalPosition().toPoint()
                delta = cur - self._press_global

                if (not self._dragging) and (abs(delta.x()) + abs(delta.y()) >= self._drag_threshold):
                    self._dragging = True

                if self._dragging:
                    self.move(self._press_win_topleft + delta)
                    return True

                return False

            if event.type() == QEvent.MouseButtonRelease:
                was_dragging = self._dragging
                self._dragging = False
                return True if was_dragging else False

        return super().eventFilter(obj, event)


def resolve_html_path(html_arg: str | None) -> Path:
    target = Path(html_arg) if html_arg else DEFAULT_HTML
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the optional desktop pet client for local HTML pages."
    )
    parser.add_argument(
        "--html",
        default=str(DEFAULT_HTML.relative_to(PROJECT_ROOT)),
        help="HTML file to load. Relative paths are resolved from the inner project root.",
    )
    parser.add_argument("--width", type=int, default=420, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=520, help="Window height in pixels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_path = resolve_html_path(args.html)
    if not html_path.is_file():
        raise SystemExit(f"HTML file not found: {html_path}")

    app = QApplication(sys.argv)
    pet = DesktopPet(html_path=html_path, width=args.width, height=args.height)
    pet.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
