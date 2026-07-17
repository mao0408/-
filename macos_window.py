from __future__ import annotations

"""macOS window discovery and activation helpers.

The module keeps Quartz/AppKit imports optional so the Windows test suite can
import the project without installing macOS-only packages.
"""

from pathlib import Path

from wechat_window import ChatPlatform, ChatWindow, WindowRect, _title_matches_platform, _process_matches_platform

try:  # pragma: no cover - exercised on macOS
    import Quartz
except Exception:  # pragma: no cover - Windows and source-only environments
    Quartz = None

try:  # pragma: no cover - exercised on macOS
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication, NSScreen, NSWorkspace
except Exception:  # pragma: no cover - Windows and source-only environments
    NSApplicationActivateIgnoringOtherApps = None
    NSRunningApplication = None
    NSScreen = None
    NSWorkspace = None


class MacChatWindowDetector:
    """Detect WeChat, WeCom, and QQ windows using macOS window metadata."""

    def foreground_chat(self) -> ChatWindow | None:
        windows = self._windows()
        front_pid = self._frontmost_pid()
        matching = [window for window in windows if window.owner_pid == front_pid]
        return _largest_window(matching)

    def window_from_hwnd(self, hwnd: int) -> ChatWindow | None:
        return next((window for window in self._windows() if window.hwnd == int(hwnd)), None)

    def any_chat_window(self) -> ChatWindow | None:
        return _largest_window(self._windows())

    def snap_geometry(self, window: ChatWindow, width: int, height: int) -> str:
        screen_width, screen_height = _main_screen_size()
        margin = 8
        x = window.rect.right + margin
        if x + width > screen_width:
            x = max(0, window.rect.left - width - margin)
        y = max(0, min(window.rect.top, screen_height - height - margin))
        return f"{width}x{height}+{x}+{y}"

    def activate(self, window: ChatWindow) -> bool:
        if NSRunningApplication is None or not window.owner_pid:
            return False
        try:
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(window.owner_pid)
            if app is None:
                return False
            options = NSApplicationActivateIgnoringOtherApps or 1
            return bool(app.activateWithOptions_(options))
        except Exception:
            return False

    def is_foreground(self, window: ChatWindow) -> bool:
        return bool(window and window.owner_pid == self._frontmost_pid())

    def _windows(self) -> list[ChatWindow]:
        if Quartz is None:
            return []
        try:
            options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
            raw_windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        except Exception:
            return []

        result: list[ChatWindow] = []
        for raw in raw_windows:
            window = self._convert_window(raw)
            if window is not None:
                result.append(window)
        return result

    def _convert_window(self, raw: dict) -> ChatWindow | None:
        owner_pid = int(raw.get(Quartz.kCGWindowOwnerPID, 0) or 0)
        title = str(raw.get(Quartz.kCGWindowName, "") or "").strip()
        owner_name = str(raw.get(Quartz.kCGWindowOwnerName, "") or "").strip()
        if not owner_name:
            owner_name = self._owner_name(owner_pid)
        platform = self._match_platform(owner_name, title)
        if platform is None:
            return None

        bounds = raw.get(Quartz.kCGWindowBounds) or {}
        rect = _rect_from_bounds(bounds)
        if rect.width < 420 or rect.height < 360:
            return None
        window_id = int(raw.get(Quartz.kCGWindowNumber, 0) or 0)
        if not window_id:
            return None
        return ChatWindow(
            hwnd=window_id,
            title=title or owner_name,
            rect=rect,
            platform=platform,
            owner_pid=owner_pid,
        )

    def _match_platform(self, owner_name: str, title: str) -> ChatPlatform | None:
        title_l = title.lower()
        process_l = Path(owner_name).name.lower()
        for profile in ChatPlatform.profiles():
            if _title_matches_platform(profile, title_l):
                return profile
            if _process_matches_platform(profile, process_l):
                return profile
        return None

    @staticmethod
    def _owner_name(pid: int) -> str:
        if NSRunningApplication is None or not pid:
            return ""
        try:
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            return str(app.localizedName() or "") if app is not None else ""
        except Exception:
            return ""

    @staticmethod
    def _frontmost_pid() -> int:
        if NSWorkspace is None:
            return 0
        try:
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return int(app.processIdentifier()) if app is not None else 0
        except Exception:
            return 0


def activate_window(window: ChatWindow) -> bool:
    return MacChatWindowDetector().activate(window)


def is_foreground_window(window: ChatWindow) -> bool:
    return MacChatWindowDetector().is_foreground(window)


def _largest_window(windows: list[ChatWindow]) -> ChatWindow | None:
    if not windows:
        return None
    return max(windows, key=lambda item: item.rect.width * item.rect.height)


def _rect_from_bounds(bounds: dict) -> WindowRect:
    origin = bounds.get("X", 0), bounds.get("Y", 0)
    size = bounds.get("Width", 0), bounds.get("Height", 0)
    left, top = int(origin[0] or 0), int(origin[1] or 0)
    width, height = int(size[0] or 0), int(size[1] or 0)
    return WindowRect(left, top, left + width, top + height)


def _main_screen_size() -> tuple[int, int]:
    if NSScreen is None:
        return 1440, 900
    try:
        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 1440, 900
