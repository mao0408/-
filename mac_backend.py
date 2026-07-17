from __future__ import annotations

"""Small local HTTP bridge for the Tauri UI.

The existing project is a Tkinter application. This bridge reuses its tested
OCR, memory, reply-engine, and sender modules without importing Tkinter.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from capture_ocr import ChatOCR
from memory import StyleMemory
from reply_engine import ReplyEngine, latest_meaningful_message, unreplied_other_messages
from sender import WeChatSender
from settings import AppSettings
from wechat_window import ChatWindow, WeChatWindowDetector


class ReplyService:
    def __init__(self) -> None:
        self.settings = AppSettings().load()
        self.memory = StyleMemory().load()
        self.detector = WeChatWindowDetector()
        self.ocr = ChatOCR(self.settings)
        self.engine = ReplyEngine(self.settings, self.memory)
        self.sender = WeChatSender()
        self.current_window: ChatWindow | None = None
        self.current_context = ""
        self.current_replies: list[str] = []
        self.lock = threading.RLock()

    def state(self) -> dict[str, Any]:
        with self.lock:
            window = self.current_window
            return {
                "platform": getattr(getattr(window, "platform", None), "key", ""),
                "window_title": getattr(window, "title", ""),
                "context": self.current_context,
                "replies": list(self.current_replies),
                "reply_source": self.engine.last_source,
                "api_ready": bool(self.settings.api_key),
                "ocr_ready": not bool(self.ocr.engine_error),
                "crm_ready": False,
                "status": "等待聊天窗口",
            }

    def recognize(self) -> dict[str, Any]:
        with self.lock:
            window = self.detector.foreground_chat() or self.detector.any_chat_window()
            if window is None:
                return {**self.state(), "status": "请先打开微信并进入聊天窗口"}
            self.current_window = window
            image = self.ocr.capture(window)
            if image is None:
                return {**self.state(), "status": self.ocr.engine_error or "无法截取聊天窗口"}
            context = self.ocr.read_image(image)
            if not context.strip():
                return {
                    **self.state(),
                    "status": "未识别到聊天文字",
                    "platform": window.platform.key,
                    "window_title": window.title,
                }
            self.current_context = context
            replies = self.engine.generate(
                context,
                window.title,
                target_turn=unreplied_other_messages(context),
            )
            self.current_replies = list(replies)
            return {
                **self.state(),
                "platform": window.platform.key,
                "window_title": window.title,
                "context": context,
                "replies": replies,
                "reply_source": self.engine.last_source,
                "status": f"已生成 {len(replies)} 条候选回复",
            }

    def generate(self, context: str, partner: str = "") -> dict[str, Any]:
        with self.lock:
            self.current_context = context.strip()
            replies = self.engine.generate(
                self.current_context,
                partner or getattr(self.current_window, "title", ""),
                target_turn=unreplied_other_messages(self.current_context),
            )
            self.current_replies = list(replies)
            return {
                **self.state(),
                "context": self.current_context,
                "replies": replies,
                "reply_source": self.engine.last_source,
                "status": f"已生成 {len(replies)} 条候选回复",
            }

    def send(self, text: str) -> dict[str, Any]:
        with self.lock:
            if self.current_window is None:
                return {**self.state(), "status": "没有可发送的聊天窗口"}
            ok, message = self.sender.send(self.current_window, text)
            if ok and text.strip():
                self.memory.learn_from_sent_reply(self.current_context, text.strip(), self.current_window.title)
                self.memory.save()
            return {**self.state(), "status": message, "sent": ok}

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            for key in (
                "api_provider",
                "api_key",
                "base_url",
                "model",
                "vision_provider",
                "vision_api_key",
                "vision_base_url",
                "vision_model",
                "reply_source_mode",
            ):
                if key in data:
                    setattr(self.settings, key, str(data[key]))
            for key in ("training_mode", "auto_recognize", "managed_auto_reply"):
                if key in data:
                    setattr(self.settings, key, bool(data[key]))
            self.settings.save()
            self.engine = ReplyEngine(self.settings, self.memory)
            return {**self.state(), "status": "设置已保存"}


class Handler(BaseHTTPRequestHandler):
    service = ReplyService()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._json({"ok": True})
            return
        if self.path == "/api/state":
            self._json(self.service.state())
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/recognize":
                self._json(self.service.recognize())
            elif self.path == "/api/generate":
                self._json(self.service.generate(str(payload.get("context", "")), str(payload.get("partner", ""))))
            elif self.path == "/api/send":
                self._json(self.service.send(str(payload.get("text", ""))))
            elif self.path == "/api/settings":
                self._json(self.service.update_settings(payload))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._headers()
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _headers(self) -> None:
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, _format: str, *_args: object) -> None:
        return


def serve(host: str = "127.0.0.1", port: int | None = None) -> None:
    port = port or int(os.environ.get("BACKEND_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    serve()
