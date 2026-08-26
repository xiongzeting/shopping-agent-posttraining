import unittest
from unittest.mock import patch

from shopping_grpo.evaluation.model_client import client_from_environment


class EvaluationModelClientTest(unittest.TestCase):
    def test_judge_profile_uses_pro_model_and_isolated_credentials(self):
        environment = {
            "SHOPPING_JUDGE_MODEL": "deepseek-v4-pro",
            "SHOPPING_JUDGE_BASE_URL": "https://judge.test/v1",
            "SHOPPING_JUDGE_API_KEY": "judge-secret",
        }
        with patch.dict("os.environ", environment, clear=False):
            client = client_from_environment(max_tokens=512)

        self.assertEqual(client.model, "deepseek-v4-pro")
        self.assertEqual(client.base_url, "https://judge.test/v1")
        self.assertEqual(client.api_key, "judge-secret")


if __name__ == "__main__":
    unittest.main()
