from __future__ import annotations

import sys
import time


try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

try:
    import pyperclip
except Exception:  # pragma: no cover
    pyperclip = None

try:
    import win32gui
    import win32con
except Exception:  # pragma: no cover
    win32gui = None
    win32con = None


class WeChatSender:
    def send(self, target: object, text: str) -> tuple[bool, str]:
        text = text.strip()
        if not text:
            return False, "发送内容为空。"
        if pyautogui is None:
            return False, "缺少 pyautogui，无法自动发送。"
        if pyperclip is None:
            return False, "缺少 pyperclip，无法写入剪贴板。"
        previous_clipboard = _read_clipboard()
        copied_for_send = False
        try:
            self._activate(target)
            time.sleep(0.15)
            if not self._is_target_foreground(target):
                return False, "目标聊天窗口没有成功切到前台，已取消发送，避免发到其他应用。"
            self._click_input(target)
            pyperclip.copy(text)
            copied_for_send = True
            time.sleep(0.08)
            pyautogui.hotkey("command" if sys.platform == "darwin" else "ctrl", "v")
            time.sleep(0.12)
            pyautogui.press("enter")
            return True, "已发送"
        except Exception as exc:
            return False, f"发送失败：{exc}"
        finally:
            if copied_for_send and previous_clipboard is not None:
                try:
                    pyperclip.copy(previous_clipboard)
                except Exception:
                    pass

    @staticmethod
    def _activate(target: object) -> None:
        if sys.platform == "darwin":
            try:
                from macos_window import activate_window

                activate_window(target)
            except Exception:
                pass
            return
        hwnd = _target_hwnd(target)
        if win32gui is None or not hwnd:
            return
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    @staticmethod
    def _click_input(target: object) -> None:
        if pyautogui is None:
            return
        rect = getattr(target, "rect", None)
        platform = getattr(target, "platform", None)
        input_point = getattr(platform, "input_point", None)
        if rect is not None and input_point is not None:
            x = rect.left + int(rect.width * input_point.x)
            y = rect.top + int(rect.height * input_point.y)
            pyautogui.click(x, y)
            return
        hwnd = _target_hwnd(target)
        if win32gui is None or not hwnd:
            return
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        pyautogui.click(left + width // 2, bottom - 58)

    @staticmethod
    def _is_target_foreground(target: object) -> bool:
        if sys.platform == "darwin":
            try:
                from macos_window import is_foreground_window

                return is_foreground_window(target)
            except Exception:
                return False
        hwnd = _target_hwnd(target)
        if win32gui is None or not hwnd:
            return True
        try:
            return int(win32gui.GetForegroundWindow() or 0) == hwnd
        except Exception:
            return False


def _target_hwnd(target: object) -> int:
    try:
        return int(getattr(target, "hwnd", target) or 0)
    except Exception:
        return 0


def _read_clipboard() -> str | None:
    try:
        return str(pyperclip.paste()) if pyperclip is not None and hasattr(pyperclip, "paste") else None
    except Exception:
        return None
