import sys
import types
import unittest


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

from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv


class _FakeBrowser:
    def __init__(self):
        self.current_url = None
        self.page_source = ""
        self.received = None

    def get(self, url, session_id=None, session_int=None):
        self.current_url = url
        self.received = {
            "url": url,
            "session_id": session_id,
            "session_int": session_int,
        }


class _FakeServer:
    def __init__(self):
        self.goals = {
            649: {
                "instruction_simple": "buy item",
                "goal_options": [],
                "user_persona": {},
            }
        }

    @staticmethod
    def resolve_goal_index(task_id):
        if task_id != 650:
            raise AssertionError(task_id)
        return 649


class TaskAlignmentTest(unittest.TestCase):
    def test_default_reset_passes_resolved_goal_index_to_server(self):
        env = WebAgentTextEnv.__new__(WebAgentTextEnv)
        env.server = _FakeServer()
        env.browser = _FakeBrowser()
        env.base_url = "http://shop.test"
        env.session_prefix = None
        env.observation_mode = "url"
        env.history_init = []
        env.if_persona = False

        env.reset(idx=650, instruction_text="buy item")

        self.assertEqual(env.session, 650)
        self.assertEqual(env.browser.received["session_id"], 650)
        self.assertEqual(env.browser.received["session_int"], 649)

    def test_unmapped_task_id_does_not_fall_back_to_goal_list_position(self):
        server = SimServer.__new__(SimServer)
        server.task_id_to_goal_index = {650: 649}

        self.assertEqual(server.resolve_goal_index(650), 649)
        with self.assertRaisesRegex(KeyError, "no aligned shopping goal"):
            server.resolve_goal_index(651)


if __name__ == "__main__":
    unittest.main()
