import json
from pathlib import Path
import unittest

from shopping_grpo.environment.tools import (
    SHOP_TOOL_SCHEMAS,
    tool_call_to_action,
)


class ShopToolsTest(unittest.TestCase):
    def test_python_and_grpo_tool_schemas_are_identical(self):
        root = Path(__file__).resolve().parents[1]
        configured = json.loads(
            (root / "configs" / "tools.json").read_text(encoding="utf-8")
        )["tools"]

        self.assertEqual(
            [entry["tool_schema"] for entry in configured],
            SHOP_TOOL_SCHEMAS,
        )

    def test_search_products_maps_to_search_action(self):
        self.assertEqual(
            tool_call_to_action("search_products", {"query": "乳胶枕"}),
            "search[乳胶枕]",
        )

    def test_buy_now_maps_to_click_action(self):
        self.assertEqual(tool_call_to_action("buy_now", {}), "click[Buy Now]")

    def test_finish_without_purchase_maps_to_explicit_terminal_action(self):
        self.assertEqual(
            tool_call_to_action(
                "finish_without_purchase",
                {"reason": "no_suitable_product"},
            ),
            "finish[no_suitable_product]",
        )

    def test_tool_schemas_expose_only_supported_tools(self):
        names = [schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS]

        self.assertEqual(
            names,
            [
                "search_products",
                "open_product",
                "select_option",
                "next_page",
                "prev_page",
                "back_to_search",
                "buy_now",
                "finish_without_purchase",
            ],
        )
        for removed_name in (
            "view_features",
            "view_description",
            "view_reviews",
            "view_attributes",
        ):
            self.assertNotIn(removed_name, names)

    def test_tool_schemas_reject_undeclared_arguments(self):
        for schema in SHOP_TOOL_SCHEMAS:
            with self.subTest(tool=schema["function"]["name"]):
                self.assertFalse(schema["function"]["parameters"]["additionalProperties"])

    def test_tool_descriptions_state_current_page_constraints(self):
        schemas = {schema["function"]["name"]: schema["function"] for schema in SHOP_TOOL_SCHEMAS}

        self.assertIn("搜索功能是否可用: True", schemas["search_products"]["description"])
        self.assertIn("不得重复相同查询", schemas["search_products"]["description"])
        self.assertIn("最新 observation", schemas["open_product"]["description"])
        self.assertIn("稳定 option_id", schemas["select_option"]["description"])
        self.assertIn("不得填写 label", schemas["select_option"]["description"])
        self.assertIn("完整 variant 的实际价格", schemas["select_option"]["description"])
        self.assertIn("Buy Now", schemas["buy_now"]["description"])
        self.assertIn("品类正确", schemas["buy_now"]["description"])
        self.assertIn("用户明确要求均已满足", schemas["buy_now"]["description"])
        self.assertIn("必须立即调用", schemas["buy_now"]["description"])
        self.assertIn("整体最符合且可接受的候选", schemas["buy_now"]["description"])
        self.assertIn("无需寻找任何预设商品", schemas["buy_now"]["description"])
        self.assertNotIn("目标 ASIN", schemas["buy_now"]["description"])
        self.assertNotIn("不可撤销", schemas["buy_now"]["description"])

    def test_finish_description_matches_single_abstain_contract(self):
        schemas = {
            schema["function"]["name"]: schema["function"]
            for schema in SHOP_TOOL_SCHEMAS
        }

        description = schemas["finish_without_purchase"]["description"]
        self.assertIn("有实质差异的搜索和候选核验", description)
        self.assertIn("没有可接受商品", description)
        self.assertIn("统一按 early_abstain 终止", description)
        self.assertNotIn("是否达到资格由环境判断", description)
        self.assertNotIn("graceful_stop", description)


if __name__ == "__main__":
    unittest.main()
