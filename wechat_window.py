from __future__ import annotations

import ctypes
from pathlib import Path
from dataclasses import dataclass

from settings import ChatArea


try:
    import win32gui
    import win32process
except Exception:  # pragma: no cover - Windows optional dependency
    win32gui = None
    win32process = None


@dataclass(frozen=True)
class InputPoint:
    x: float
    y: float


@dataclass(frozen=True)
class ChatPlatform:
    key: str
    label: str
    title_keywords: tuple[str, ...]
    process_keywords: tuple[str, ...]
    class_keywords: tuple[str, ...]
    chat_area: ChatArea
    input_point: InputPoint

    @classmethod
    def profiles(cls) -> tuple["ChatPlatform", ...]:
        return (
            cls(
                key="wechat",
                label="微信",
                title_keywords=("微信", "WeChat", "Weixin"),
                process_keywords=("wechat", "weixin"),
                class_keywords=("wechat", "qt5"),
                chat_area=ChatArea(0.31, 0.98, 0.09, 0.91),
                input_point=InputPoint(0.66, 0.92),
            ),
            cls(
                key="wecom",
                label="企业微信",
                title_keywords=("企业微信", "WeCom", "WXWork"),
                process_keywords=("wxwork", "wecom"),
                class_keywords=("wxwork", "wecom", "qt5"),
                chat_area=ChatArea(0.28, 0.74, 0.10, 0.88),
                input_point=InputPoint(0.48, 0.88),
            ),
            cls(
                key="qq",
                label="QQ",
                title_keywords=("QQ",),
                process_keywords=("qq.exe", "ntqq", "qqnt", "qq"),
                class_keywords=("txguifoundation", "qq"),
                chat_area=ChatArea(0.36, 0.98, 0.12, 0.68),
                input_point=InputPoint(0.68, 0.94),
            ),
        )

    @classmethod
    def for_key(cls, key: str) -> "ChatPlatform":
        for profile in cls.profiles():
            if profile.key == key:
                return profile
        raise KeyError(key)

@dataclass
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class ChatWindow:
    hwnd: int
    title: str
    rect: WindowRect
    platform: ChatPlatform
    owner_pid: int = 0

    @property
    def display_title(self) -> str:
        return f"{self.platform.label}：{self.title}"


class ChatWindowDetector:
    def foreground_chat(self) -> ChatWindow | None:
        if win32gui is None:
            return None
        hwnd = win32gui.GetForegroundWindow()
        return self._window_from_hwnd(hwnd)

    def window_from_hwnd(self, hwnd: int) -> ChatWindow | None:
        return self._window_from_hwnd(hwnd)

    def any_chat_window(self) -> ChatWindow | None:
        if win32gui is None:
            return None
        candidates: list[ChatWindow] = []

        def collect(hwnd: int, _extra: object) -> bool:
            window = self._window_from_hwnd(hwnd)
            if window:
                candidates.append(window)
            return True

        win32gui.EnumWindows(collect, None)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.rect.width * item.rect.height, reverse=True)
        return candidates[0]

    def _window_from_hwnd(self, hwnd: int) -> ChatWindow | None:
        if win32gui is None:
            return None
        if not hwnd or not win32gui.IsWindowVisible(hwnd):
            return None
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return None
        platform = self._match_platform(hwnd, title)
        if platform is None:
            return None
        raw = win32gui.GetWindowRect(hwnd)
        rect = WindowRect(*raw)
        if rect.width < 420 or rect.height < 360:
            return None
        return ChatWindow(hwnd=hwnd, title=title, rect=rect, platform=platform)

    def snap_geometry(self, window: ChatWindow, width: int, height: int) -> str:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        margin = 8
        x = window.rect.right + margin
        if x + width > screen_w:
            x = max(0, window.rect.left - width - margin)
        y = max(0, min(window.rect.top, screen_h - height - margin))
        return f"{width}x{height}+{x}+{y}"

    def _match_platform(self, hwnd: int, title: str) -> ChatPlatform | None:
        process_name = self._process_name(hwnd)
        class_name = self._class_name(hwnd)
        title_l = title.lower()
        process_l = process_name.lower()
        class_l = class_name.lower()
        for profile in ChatPlatform.profiles():
            if _title_matches_platform(profile, title_l):
                return profile
            if _process_matches_platform(profile, process_l):
                return profile
            if any(keyword.lower() in class_l for keyword in profile.class_keywords):
                return profile
        return None

    @staticmethod
    def _class_name(hwnd: int) -> str:
        try:
            return win32gui.GetClassName(hwnd) if win32gui is not None else ""
        except Exception:
            return ""

    @staticmethod
    def _process_name(hwnd: int) -> str:
        if win32process is None:
            return ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return ""
        return ChatWindowDetector._pid_process_name(pid)

    @staticmethod
    def _pid_process_name(pid: int) -> str:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(len(buf))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value
            return ""
        finally:
            kernel32.CloseHandle(handle)


WeChatWindow = ChatWindow
if __import__("sys").platform == "darwin":
    class WeChatWindowDetector:  # type: ignore[no-redef]
        def __new__(cls, *args, **kwargs):
            from macos_window import MacChatWindowDetector

            return MacChatWindowDetector(*args, **kwargs)
else:
    WeChatWindowDetector = ChatWindowDetector


def _title_matches_platform(profile: ChatPlatform, title_l: str) -> bool:
    if profile.key == "qq":
        cleaned = title_l.strip()
        return cleaned == "qq" or cleaned.startswith("qq ")
    return any(keyword.lower() in title_l for keyword in profile.title_keywords)


def _process_matches_platform(profile: ChatPlatform, process_l: str) -> bool:
    if not process_l:
        return False
    name = Path(process_l).name.lower()
    stem = Path(name).stem.lower()
    for keyword in profile.process_keywords:
        key = keyword.lower()
        key_name = Path(key).name.lower()
        key_stem = Path(key_name).stem.lower()
        if name == key_name or stem == key_stem:
            return True
    return False
