import unittest
from unittest.mock import patch

import mac_backend


class MacBackendTests(unittest.TestCase):
    def test_state_keeps_current_reply_candidates(self):
        service = mac_backend.ReplyService()
        service.current_context = "[对方] 你好"
        service.current_replies = ["收到，我看一下"]

        state = service.state()

        self.assertEqual(state["context"], "[对方] 你好")
        self.assertEqual(state["replies"], ["收到，我看一下"])

    def test_generate_updates_context_and_candidates(self):
        service = mac_backend.ReplyService()
        with patch.object(service.engine, "generate", return_value=["候选回复"]) as generate:
            result = service.generate("[对方] 请问什么时候发货", "客户")

        generate.assert_called_once()
        self.assertEqual(result["context"], "[对方] 请问什么时候发货")
        self.assertEqual(result["replies"], ["候选回复"])
        self.assertEqual(service.current_replies, ["候选回复"])


if __name__ == "__main__":
    unittest.main()
