import sys
import types
import unittest
from unittest.mock import Mock


try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    gym_module = types.ModuleType("gym")
    gym_module.Env = object
    gym_envs = types.ModuleType("gym.envs")
    gym_registration = types.ModuleType("gym.envs.registration")
    gym_registration.register = lambda **_kwargs: None
    sys.modules["gym"] = gym_module
    sys.modules["gym.envs"] = gym_envs
    sys.modules["gym.envs.registration"] = gym_registration

from web_agent_site.envs.web_agent_text_env import SimServer


class CandidateRecoveryTest(unittest.TestCase):
    def _server(self):
        server = SimServer.__new__(SimServer)
        server.product_item_dict = {
            "111111111111": {"asin": "111111111111"},
            "222222222222": {"asin": "222222222222"},
        }
        server.user_sessions = {
            "session": {
                "asin": "222222222222",
                "asins": {"111111111111", "222222222222"},
                "keywords": ["current"],
                "page": 3,
                "options": {"color": "red"},
                "candidate_locations": {
                    "111111111111": {"keywords": ["saved", "query"], "page": 2}
                },
                "candidate_options": {
                    "111111111111": {"size": "large"},
                    "222222222222": {"color": "red"},
                },
                "current_page_asins": ["222222222222"],
                "subpage": "features",
            }
        }
        server._render_item_page = Mock(return_value=("html", "url"))
        return server

    def test_reopen_restores_saved_public_candidate_state(self):
        server = self._server()

        result = server.reopen_candidate("session", "111111111111")

        session = server.user_sessions["session"]
        self.assertEqual(result, ("html", "url"))
        self.assertEqual(session["asin"], "111111111111")
        self.assertEqual(session["keywords"], ["saved", "query"])
        self.assertEqual(session["page"], 2)
        self.assertEqual(session["options"], {"size": "large"})
        self.assertEqual(session["current_page_asins"], [])
        self.assertIsNone(session["subpage"])

    def test_reopen_rejects_product_not_opened_in_current_session(self):
        server = self._server()

        with self.assertRaisesRegex(ValueError, "not opened"):
            server.reopen_candidate("session", "999999999999")


if __name__ == "__main__":
    unittest.main()
