import unittest
from unittest.mock import patch

import macos_window
from wechat_window import ChatPlatform, WindowRect


class MacWindowAdapterTests(unittest.TestCase):
    def test_bounds_are_converted_to_window_rect(self):
        rect = macos_window._rect_from_bounds({"X": 12, "Y": 34, "Width": 900, "Height": 600})

        self.assertEqual(rect, WindowRect(12, 34, 912, 634))

    def test_largest_window_prefers_chat_area(self):
        small = type("Window", (), {"rect": WindowRect(0, 0, 500, 400)})()
        large = type("Window", (), {"rect": WindowRect(0, 0, 1200, 800)})()

        self.assertIs(macos_window._largest_window([small, large]), large)

    def test_adapter_returns_no_windows_without_quartz(self):
        detector = macos_window.MacChatWindowDetector()

        with patch.object(macos_window, "Quartz", None):
            self.assertIsNone(detector.foreground_chat())
            self.assertIsNone(detector.any_chat_window())

    def test_platform_profiles_match_mac_app_names(self):
        detector = macos_window.MacChatWindowDetector()

        self.assertEqual(detector._match_platform("WeChat", "张三") .key, "wechat")
        self.assertEqual(detector._match_platform("WeCom", "客户群") .key, "wecom")
        self.assertEqual(detector._match_platform("QQ", "客户") .key, "qq")


if __name__ == "__main__":
    unittest.main()
