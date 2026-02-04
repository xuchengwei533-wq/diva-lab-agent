import os
import sys
from PySide6.QtCore import Qt, QUrl, QPoint, QEvent
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView


class DesktopPet(QWidget):
    def __init__(self, html_path: str):
        super().__init__()

        # 1) 无边框 + 置顶 + 工具窗（通常不出现在任务栏）
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )

        # 2) 透明窗口
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # 你可以按需要调窗口尺寸（与模型 scale 配合）
        self.resize(420, 520)

        # 3) WebEngine 载入你的 mao-demo.html
        self.web = QWebEngineView(self)
        self.web.setAttribute(Qt.WA_TranslucentBackground, True)
        self.web.page().setBackgroundColor(Qt.transparent)

        # 关键：捕获 webview 鼠标事件用于拖动
        self.web.installEventFilter(self)

        url = QUrl.fromLocalFile(os.path.abspath(html_path))
        self.web.load(url)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web)
        self.setLayout(layout)

        # ===== 拖动状态 =====
        self._press_global = QPoint()
        self._press_win_topleft = QPoint()
        self._dragging = False
        self._drag_threshold = 6  # 像素阈值：超过才判定为拖动

    def eventFilter(self, obj, event):
        if obj is self.web:
            # 记录按下位置（不拦截，让网页仍可接收 click）
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._press_global = event.globalPosition().toPoint()
                self._press_win_topleft = self.frameGeometry().topLeft()
                self._dragging = False
                return False  # 不拦截：保证网页点击可用

            # 鼠标移动：超过阈值才开始拖动窗口
            if event.type() == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
                cur = event.globalPosition().toPoint()
                delta = cur - self._press_global

                if (not self._dragging) and (abs(delta.x()) + abs(delta.y()) >= self._drag_threshold):
                    self._dragging = True

                if self._dragging:
                    self.move(self._press_win_topleft + delta)
                    return True  # 拦截移动：避免页面滚动/选中等干扰

                return False

            # 松开：结束拖动（不拦截松开，让网页仍可正常完成一次点击）
            if event.type() == QEvent.MouseButtonRelease:
                was_dragging = self._dragging
                self._dragging = False
                # 如果刚才在拖动，拦截释放可减少“拖完触发点击”的概率
                return True if was_dragging else False

        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 改成你的实际路径：mao-demo.html
    pet = DesktopPet(html_path=r"renderer\mao-demo.html")
    pet.show()

    sys.exit(app.exec())
