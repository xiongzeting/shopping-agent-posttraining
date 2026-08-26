import json
import unittest
from pathlib import Path

from web_agent_site.engine.config import (
    load_config,
    validate_config,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "environment.json"


class EnvironmentV24ConfigTest(unittest.TestCase):
    def test_repository_config_matches_reward_contract(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["environment_version"],
            "shopsimulator-environment-v2.4",
        )
        self.assertEqual(config["reward"]["wrong_purchase"], -1.0)
        self.assertEqual(config["reward"]["assistant_final"], -0.8)
        self.assertEqual(config["reward"]["early_abstain"], -0.4)
        self.assertEqual(config["reward"]["max_steps"], 0.0)
        self.assertEqual(config["reward"]["repeat_loop"], -0.6)
        self.assertEqual(config["reward"]["partial_purchase_base"], 0.5)
        self.assertEqual(config["reward"]["partial_purchase_scale"], 0.3)
        self.assertEqual(config["reward"]["version"], "shopsimulator-reward-v4")
        self.assertEqual(
            config["reward_feature_version"],
            "shopping-reward-features-v2",
        )
        self.assertEqual(
            config["termination"]["version"],
            "shopping-termination-v3.2",
        )
        self.assertEqual(config["termination"]["no_progress_limit"], 6)

    def test_reward_drift_is_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["reward"]["wrong_purchase"] = -0.4
        with self.assertRaisesRegex(ValueError, "reward values"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
