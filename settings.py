from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


APP_NAME = "WeChatStyleReplyAssistant"


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_data_dir() -> Path:
    return app_base_dir() / "user_data"


def default_config_path() -> Path:
    return app_data_dir() / "settings.json"


@dataclass
class ChatArea:
    x_left: float = 0.31
    x_right: float = 0.98
    y_top: float = 0.09
    y_bottom: float = 0.91


@dataclass
class AppSettings:
    config_path: Path = field(default_factory=default_config_path)
    api_provider: str = "zhipu"
    api_key: str = ""
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-4-flash-250414"
    vision_provider: str = "zhipu"
    vision_api_key: str = ""
    vision_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    vision_model: str = "glm-4v-flash"
    custom_headers: str = '{"Authorization":"Bearer {api_key}","Content-Type":"application/json"}'
    custom_body: str = '{"model":"{model}","messages":{messages_json}}'
    custom_response_path: str = "choices.0.message.content"
    reply_source_mode: str = "model_memory"
    ocr_engine: str = "rapidocr"
    training_mode: bool = False
    auto_recognize: bool = False
    managed_auto_reply: bool = False
    auto_reply_delay_seconds: float = 1.2
    poll_interval_ms: int = 1200
    capture_cooldown_seconds: float = 4.0
    chat_area: ChatArea = field(default_factory=ChatArea)

    def load(self) -> "AppSettings":
        if not self.config_path.exists():
            env_key = os.environ.get("TEXT_API_KEY", "") or os.environ.get("ZHIPU_API_KEY", "")
            if env_key:
                self.api_key = env_key
            return self

        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.api_provider = str(data.get("api_provider", self.api_provider))
        self.api_key = str(data.get("api_key", self.api_key))
        self.base_url = str(data.get("base_url", self.base_url))
        self.model = str(data.get("model", self.model))
        self.vision_provider = str(data.get("vision_provider", self.vision_provider))
        self.vision_api_key = str(data.get("vision_api_key", self.vision_api_key))
        self.vision_base_url = str(data.get("vision_base_url", self.vision_base_url))
        self.vision_model = str(data.get("vision_model", self.vision_model))
        if self.api_provider == "zhipu" and self.model in {"glm-4.7-flash", "glm-4-flash"}:
            self.model = "glm-4-flash-250414"
        self.custom_headers = str(data.get("custom_headers", self.custom_headers))
        self.custom_body = str(data.get("custom_body", self.custom_body))
        self.custom_response_path = str(data.get("custom_response_path", self.custom_response_path))
        self.reply_source_mode = str(data.get("reply_source_mode", self.reply_source_mode))
        if self.reply_source_mode not in {"model", "memory", "model_memory"}:
            self.reply_source_mode = "model_memory"
        self.ocr_engine = str(data.get("ocr_engine", self.ocr_engine))
        self.training_mode = bool(data.get("training_mode", self.training_mode))
        self.auto_recognize = bool(data.get("auto_recognize", self.auto_recognize))
        self.managed_auto_reply = bool(data.get("managed_auto_reply", self.managed_auto_reply))
        self.auto_reply_delay_seconds = float(data.get("auto_reply_delay_seconds", self.auto_reply_delay_seconds))
        self.poll_interval_ms = int(data.get("poll_interval_ms", self.poll_interval_ms))
        self.capture_cooldown_seconds = float(
            data.get("capture_cooldown_seconds", self.capture_cooldown_seconds)
        )
        area = data.get("chat_area", {})
        if isinstance(area, dict):
            self.chat_area = ChatArea(
                x_left=float(area.get("x_left", self.chat_area.x_left)),
                x_right=float(area.get("x_right", self.chat_area.x_right)),
                y_top=float(area.get("y_top", self.chat_area.y_top)),
                y_bottom=float(area.get("y_bottom", self.chat_area.y_bottom)),
            )
            if self.chat_area.x_left < 0.25:
                self.chat_area.x_left = ChatArea().x_left
                self.chat_area.x_right = ChatArea().x_right
                self.chat_area.y_top = ChatArea().y_top
                self.chat_area.y_bottom = ChatArea().y_bottom
            if self.chat_area.y_bottom < 0.90:
                self.chat_area.y_bottom = ChatArea().y_bottom
        return self

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["config_path"] = str(self.config_path)
        self.config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
